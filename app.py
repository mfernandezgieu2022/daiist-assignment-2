"""
Assignment 2: dashboard.

Loads the committed weights (weights/best.pt) and the files train.py saved in
artifacts/. It never trains anything. Run with: uv run python main.py app

Tabs
  1. Try an image        upload an image, see the boxes and the PASS / REVIEW / FAIL decision
  2. Thresholds and cost move the two thresholds, see the confusion matrix, the expected
                         cost per 1,000 cases, the review load and the cost-vs-threshold curve
  3. Per-class results   precision, recall, AP, confusion matrix, baselines
"""
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import costing  # noqa: E402

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "weights" / "best.pt"
ART = ROOT / "artifacts"
T = costing.THRESHOLDS


# ------------------------------ load saved outputs -------------------------
def _load(name: str) -> dict:
    path = ART / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run: uv run python main.py train")
    return json.loads(path.read_text())


if not WEIGHTS.exists():
    raise FileNotFoundError(f"{WEIGHTS} is missing. Train (or copy) your weights first.")

from ultralytics import YOLO  # noqa: E402

MODEL = YOLO(str(WEIGHTS))
METRICS = _load("metrics.json")
SCORES = _load("test_scores.json")
SWEEP = _load("cost_sweep.json")
POINT = _load("operating_point.json")
A = costing.Assumptions(**SWEEP["assumptions"])
POSITIVE = SCORES["positive_class"]
ROWS = SCORES["images"]
SETS = {"Clean test (141 images, leaked images removed)": True,
        "Full test (435 images, many near-duplicates of train)": False}
DEFAULT_SET = list(SETS)[0]
BASE_RATES = [0.0001, 0.0003, 0.001, 0.003, 0.01]


def _idx(x: float) -> int:
    return int(round(float(x) * 100))


def fire_note(fire_score: float, outcome: str, fail_at: float) -> str:
    """The decision is about smoke only. Say so when a clear fire would otherwise pass."""
    if outcome == "FAIL" or fire_score < max(fail_at, 0.25):
        return ""
    return (f"\n\n⚠️ **A fire is visible (confidence {fire_score:.2f}).** This triage decision is about "
            f"smoke only, so a fire-only scene is not flagged here. That is a known limitation "
            f"of this design: a visible fire would need its own rule.")


def decide(score: float, a: float, b: float) -> str:
    if score >= b:
        return "FAIL"
    return "REVIEW" if score >= a else "PASS"


# ------------------------------ tab 1: one image --------------------------
def predict_image(image, pass_below, fail_at):
    if image is None:
        return None, "Upload an image to see predictions."
    a, b = min(pass_below, fail_at), fail_at
    r = MODEL.predict(image, conf=0.001, iou=0.6, imgsz=640, max_det=100, verbose=False)[0]
    names = MODEL.names
    cls, conf = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
    best = {names[c]: float(conf[cls == c].max()) if (cls == c).any() else 0.0 for c in names}
    shown = r[conf >= max(a, 0.01)] if len(conf) else r
    drawn = shown.plot()[..., ::-1]                      # BGR -> RGB
    smoke = best.get(POSITIVE, 0.0)
    outcome = decide(smoke, a, b)
    badge = {"PASS": "🟢 **PASS**: no smoke alert",
             "REVIEW": "🟠 **SEND FOR HUMAN REVIEW**",
             "FAIL": "🔴 **FAIL**: raise a smoke alert"}[outcome]
    text = (f"{badge}\n\nBest **{POSITIVE}** confidence: **{smoke:.2f}** "
            f"(pass below {a:.2f}, fail at or above {b:.2f}).\n\n"
            f"Best fire confidence: {best.get('Fire', 0.0):.2f} (shown for information; "
            f"the decision uses smoke only). Boxes below {max(a, 0.01):.2f} are hidden."
            + fire_note(best.get("Fire", 0.0), outcome, b))
    return drawn, text


# ------------------------------ tab 2: thresholds and cost ----------------
def _subset(which: str):
    only_clean = SETS[which]
    rows = [r for r in ROWS if not (only_clean and r.get("leaky"))]
    pos, neg = costing._split(ROWS, POSITIVE, only_clean=only_clean)
    return rows, pos, neg


def update_cost(pass_below, fail_at, which, base_rate):
    a_thr, b_thr = min(pass_below, fail_at), fail_at
    i, j = _idx(a_thr), _idx(b_thr)
    rows, pos, neg = _subset(which)
    pi = float(base_rate)

    # 1) measured counts on the test images
    counts = {("Smoke present", "PASS"): 0, ("Smoke present", "REVIEW"): 0, ("Smoke present", "FAIL"): 0,
              ("No smoke", "PASS"): 0, ("No smoke", "REVIEW"): 0, ("No smoke", "FAIL"): 0}
    for r in rows:
        actual = "Smoke present" if POSITIVE in r["gt_classes"] else "No smoke"
        counts[(actual, decide(r["scores"][POSITIVE], a_thr, b_thr))] += 1
    table = pd.DataFrame(
        [[counts[(k, o)] for o in ("PASS", "REVIEW", "FAIL")] for k in ("Smoke present", "No smoke")],
        index=["Smoke present (real)", "No smoke (real)"], columns=["PASS", "REVIEW", "FAIL"],
    ).reset_index(names="")

    # 2) expected cost at the assumed base rate (not the test set's smoke share)
    tpr, fpr = costing.rates(pos, neg)
    cost, share = costing.cost_grid(tpr, fpr, A, pi)
    here = float(cost[i, j])
    load = costing.daily_load(A, float(share[i, j]), tpr[i], tpr[j], fpr[i], fpr[j])
    single = np.diag(cost)
    k = int(np.argmin(single))
    nothing, everything = costing.cost_do_nothing(A, pi), costing.cost_review_everything(A, pi)
    n_test = len(rows)
    review_test = (counts[("Smoke present", "REVIEW")] + counts[("No smoke", "REVIEW")]) / max(n_test, 1)

    md = (
        f"### Expected cost per 1,000 cases: **€{1000 * here:,.0f}**\n"
        f"at an assumed base rate of {pi:.2%} smoke (the test set itself is "
        f"{len(pos) / max(len(pos) + len(neg), 1):.0%} smoke, which is not realistic).\n\n"
        f"| Strategy | Cost per 1,000 cases |\n|---|---|\n"
        f"| **Your thresholds** ({a_thr:.2f} / {b_thr:.2f}) | **€{1000 * here:,.0f}** |\n"
        f"| Do nothing | €{1000 * nothing:,.0f} |\n"
        f"| A person checks everything | €{1000 * everything:,.0f} |\n"
        f"| Model alone, best single threshold ({T[k]:.2f}) | €{1000 * single[k]:,.0f} |\n\n"
        f"### Review load\n"
        f"- Share sent for review at the assumed base rate: **{load['review_share']:.1%}** "
        f"(on the test images as they are: {review_test:.1%})\n"
        f"- At {load['cases_per_day']:,} snapshots a day ({A.cameras} cameras, one every "
        f"{A.snapshot_minutes:g} min): **{load['reviews_per_day']:,} reviews a day**, "
        f"{load['reviewer_hours_per_day']} reviewer-hours at {A.review_seconds:g} s each\n"
        f"- That uses **{load['capacity_used']:.0%}** of the {A.reviewers_per_shift} reviewer(s) per shift "
        f"available; about **{load['people_on_roster']} people** on the roster for 24/7 cover\n"
        f"- Alerts a day: {load['true_alerts_per_day']} real, **{load['false_alerts_per_day']} false**\n"
        + ("\n⚠️ Review load is above what the available people can do.\n"
           if load["capacity_used"] > 1 else "")
        + ("\nNote: pass-below was above fail-at, so it was set equal to fail-at (no review band).\n"
           if pass_below > fail_at else "")
    )

    # 3) cost vs threshold curve
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(T, 1000 * single, label="model decides alone")
    ax.axhline(1000 * nothing, ls="--", c="grey", label="do nothing")
    ax.axhline(1000 * here, ls=":", c="C2", label=f"your thresholds: €{1000 * here:,.0f}")
    ax.axvspan(a_thr, b_thr, color="C2", alpha=0.2, label=f"your review band {a_thr:.2f}-{b_thr:.2f}")
    shipped = POINT["single_threshold"]
    ax.plot([shipped], [1000 * single[_idx(shipped)]], "r*", ms=15,
            label=f"chosen single threshold {shipped:.2f}")
    ax.set_yscale("log")   # costs at very low thresholds are huge; log keeps the minimum visible
    ax.set_xlabel("confidence threshold (smoke)")
    ax.set_ylabel("expected cost per 1,000 cases (EUR)")
    ax.set_title(f"Cost vs threshold (base rate {pi:.2%})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return table, md, fig


# ------------------------------ tab 3: per-class results -------------------
def _per_class_table() -> pd.DataFrame:
    rows = []
    for key, label in (("clean_test", "Clean test"), ("full_test", "Full test")):
        d = METRICS[key]["detection"]
        n = METRICS[key]["n_images"]
        for name, v in d["per_class"].items():
            rows.append([f"{label} ({n})", name, v["precision"], v["recall"], v["ap50"], v["map50_95"]])
        o = d["overall"]
        rows.append([f"{label} ({n})", "ALL", o["precision"], o["recall"], o["ap50"], o["map50_95"]])
    return pd.DataFrame(rows, columns=["Test set (images)", "Class", "Precision", "Recall", "AP50", "mAP50-95"])


def _confusion_table(key: str) -> pd.DataFrame:
    cm = METRICS[key]["confusion_matrix"]
    labels = cm["labels"]
    df = pd.DataFrame(cm["matrix"], columns=[f"true {x}" for x in labels],
                      index=[f"predicted {x}" for x in labels])
    return df.reset_index(names="")


def _baseline_table() -> pd.DataFrame:
    b = METRICS["baselines"]
    rows = []
    for key, label in (("clean_test", "Finetuned YOLO11s (clean test)"), ("full_test", "Finetuned YOLO11s (full test)")):
        il = METRICS[key]["image_level"]
        rows.append([label, METRICS[key]["detection"]["per_class"][POSITIVE]["ap50"],
                     il["recall"], il["precision"]])
    rows.append(["Pretrained COCO, zero-shot", 0.0, 0.0, None])
    rows.append(["Predict nothing", 0.0, b["predict_nothing"]["recall"], b["predict_nothing"]["precision"]])
    rows.append(["Always alert", None, b["always_alert"]["recall"], b["always_alert"]["precision"]])
    return pd.DataFrame(rows, columns=[
        "Model", f"{POSITIVE} AP50", f"{POSITIVE} recall (image level, 0.25)",
        f"{POSITIVE} precision (image level, 0.25)"])


def _assumptions_md() -> str:
    a = A
    return (
        "| Assumption | Value |\n|---|---|\n"
        f"| Missed smoke (false negative) | €{a.cost_fn:,.0f} (5% chance it becomes a big fire × €100,000) |\n"
        f"| False alarm (false positive) | €{a.cost_fp:,.0f} (patrol: 2 staff × 1.5 h × €35 + vehicle) |\n"
        f"| One human review | €{a.cost_review:.2f} (about {a.review_seconds:g} s of operator time) |\n"
        f"| Correct alert / correct pass | €{a.cost_tp:g} / €{a.cost_tn:g} |\n"
        f"| Reviewer misses real smoke / confirms false smoke | {a.reviewer_miss:.0%} / {a.reviewer_false_alarm:.0%} |\n"
        f"| Base rate (snapshots with smoke) | {a.base_rate:.2%} |\n"
        f"| Network | {a.cameras} cameras, one snapshot every {a.snapshot_minutes:g} min |\n"
        f"| Reviewers available | {a.reviewers_per_shift} per shift, {a.shifts_per_day} shifts, "
        f"{a.utilisation:.0%} of each shift spent reviewing |"
    )


# ------------------------------ layout ------------------------------------
def build() -> gr.Blocks:
    with gr.Blocks(title="Smoke and fire alert triage") as demo:
        gr.Markdown(
            "# Smoke and fire alert triage\n"
            f"A YOLO11s model finetuned on Roboflow `sayed-gamall/fire-smoke-detection-yolov11` v2. "
            f"Decision: **PASS**, **FAIL** (smoke alert) or **send for human review**, based on the "
            f"model's best **{POSITIVE}** confidence."
        )
        with gr.Row():
            pass_below = gr.Slider(0, 1, value=POINT["pass_below"], step=0.01,
                                   label="Pass below (scores under this are PASS)")
            fail_at = gr.Slider(0, 1, value=POINT["fail_at_or_above"], step=0.01,
                                label="Fail at or above (scores from here are FAIL; in between = REVIEW)")

        with gr.Tab("Try an image"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Upload an image")
                img_out = gr.Image(label="Predictions")
            verdict = gr.Markdown()
            for comp in (img_in, pass_below, fail_at):
                comp.change(predict_image, [img_in, pass_below, fail_at], [img_out, verdict])

        with gr.Tab("Thresholds and cost"):
            with gr.Row():
                which = gr.Radio(list(SETS), value=DEFAULT_SET, label="Evaluate on")
                base = gr.Dropdown(BASE_RATES, value=A.base_rate, label="Assumed base rate (share of snapshots with smoke)")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Confusion matrix (test images, counted)")
                    cm = gr.Dataframe(interactive=False)
                    summary = gr.Markdown()
                plot = gr.Plot(label="Cost vs threshold")
            for comp in (pass_below, fail_at, which, base):
                comp.change(update_cost, [pass_below, fail_at, which, base], [cm, summary, plot])
            demo.load(update_cost, [pass_below, fail_at, which, base], [cm, summary, plot])
            with gr.Accordion("Cost assumptions (all figures are assumptions)", open=False):
                gr.Markdown(_assumptions_md())

        with gr.Tab("Per-class results"):
            gr.Markdown(
                "Detection metrics on the test split. **Precision and recall are at the "
                "F1-best confidence.** The clean test removes test images that are near-duplicates "
                "of train or valid images, so it is the honest one, but it is small."
            )
            gr.Dataframe(_per_class_table(), interactive=False, label="Per class")
            with gr.Row():
                gr.Dataframe(_confusion_table("clean_test"), interactive=False,
                             label="Confusion matrix, clean test (conf 0.25; rows predicted, columns true)")
                gr.Dataframe(_confusion_table("full_test"), interactive=False,
                             label="Confusion matrix, full test (conf 0.25)")
            gr.Dataframe(_baseline_table(), interactive=False, label="Against the baselines")
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()