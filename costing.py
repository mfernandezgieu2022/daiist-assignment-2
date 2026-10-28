"""
Costing and human-review analysis.  Pure numpy: used by train.py (to write the
results) and by app.py (to recompute costs live as the sliders move).

THE DECISION. For every camera snapshot ("case") the model gives a score = its
best SMOKE confidence in the image. Two thresholds turn that score into three
outcomes:
    score <  pass_below            -> PASS    (no alert)
    pass_below <= score < fail_at  -> REVIEW  (a person looks at the image)
    score >= fail_at               -> FAIL    (alert: send a patrol)
With pass_below == fail_at there is no review band: the model decides alone.

EVERY FIGURE BELOW IS AN ASSUMPTION. Justify each one in REPORT.md and say what
would change the answer. `run_costing` also prints how the result moves when the
important ones change.
"""
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


@dataclass
class Assumptions:
    # ---- costs in euros per case (one camera snapshot) --------------------
    # Missed smoke: 5% chance an undetected early plume grows into a significant
    # wildfire x EUR 100,000 average suppression + damage cost.
    cost_fn: float = 5000.0
    # False alarm: patrol of 2 staff x 1.5 h x EUR 35/h (= 105) + vehicle (= 45).
    cost_fp: float = 150.0
    # One human check: ~30 s of operator time at EUR 35/h (= 0.29) + overhead.
    cost_review: float = 0.50
    # A real alert sends a patrol that is needed anyway, and a correct "pass"
    # costs nothing, so neither carries an incremental cost.
    cost_tp: float = 0.0
    cost_tn: float = 0.0
    # Share of snapshots that really contain smoke. NOT the test set's 37%.
    base_rate: float = 0.001
    # A reviewer looking at one image is not perfect.
    reviewer_miss: float = 0.05          # waves through 5% of real smoke
    reviewer_false_alarm: float = 0.02   # confirms smoke in 2% of non-smoke
    # ---- operations -------------------------------------------------------
    cameras: int = 40
    snapshot_minutes: float = 10.0       # one image per camera every 10 minutes
    review_seconds: float = 30.0         # time to check one image
    reviewers_per_shift: int = 1         # operators the agency can put on review
    shifts_per_day: int = 3              # 24/7 coverage, 8 h shifts
    shift_hours: float = 8.0
    utilisation: float = 0.75            # share of a shift spent actually reviewing

    @property
    def cases_per_day(self) -> float:
        return self.cameras * 24 * 60 / self.snapshot_minutes

    @property
    def review_capacity_per_day(self) -> float:
        """Reviews the available people can do per day."""
        secs = self.reviewers_per_shift * self.shifts_per_day * self.shift_hours * 3600
        return secs * self.utilisation / self.review_seconds

    @property
    def review_cap_share(self) -> float:
        return min(1.0, self.review_capacity_per_day / self.cases_per_day)


THRESHOLDS = np.round(np.linspace(0, 1, 101), 2)


# ------------------------------ core maths ---------------------------------
def rates(pos_scores, neg_scores, thresholds=THRESHOLDS):
    """TPR(t) and FPR(t): the share of real-smoke / no-smoke images with score >= t.
    Smoothed with +0.5/+1 (Jeffreys) so a rate is never exactly 0 or 1: with only
    a few dozen negatives, "0 false alarms" really means "fewer than about 1 in n"."""
    pos, neg = np.sort(np.asarray(pos_scores)), np.sort(np.asarray(neg_scores))
    k_pos = len(pos) - np.searchsorted(pos, thresholds, side="left")
    k_neg = len(neg) - np.searchsorted(neg, thresholds, side="left")
    return (k_pos + 0.5) / (len(pos) + 1), (k_neg + 0.5) / (len(neg) + 1)


def cost_grid(tpr, fpr, a: Assumptions, base_rate=None):
    """Expected cost per case and review share for every (pass_below, fail_at)
    pair. Rows = pass_below index, columns = fail_at index. Invalid pairs
    (pass_below > fail_at) are inf / nan."""
    pi = a.base_rate if base_rate is None else base_rate
    ta, tb = tpr[:, None], tpr[None, :]
    fa, fb = fpr[:, None], fpr[None, :]
    p_pass, p_alert, p_rev = 1 - ta, tb, ta - tb      # real smoke
    n_pass, n_alert, n_rev = 1 - fa, fb, fa - fb      # no smoke
    m, f = a.reviewer_miss, a.reviewer_false_alarm
    pos = (p_pass * a.cost_fn + p_alert * a.cost_tp
           + p_rev * (a.cost_review + m * a.cost_fn + (1 - m) * a.cost_tp))
    neg = (n_alert * a.cost_fp + n_pass * a.cost_tn
           + n_rev * (a.cost_review + f * a.cost_fp + (1 - f) * a.cost_tn))
    valid = np.triu(np.ones((len(tpr), len(tpr)), dtype=bool))
    cost = pi * pos + (1 - pi) * neg
    share = pi * p_rev + (1 - pi) * n_rev
    return np.where(valid, cost, np.inf), np.where(valid, share, np.nan)


def single_curve(tpr, fpr, a: Assumptions, base_rate=None):
    """Cost per case if the model decides alone at each threshold."""
    cost, _ = cost_grid(tpr, fpr, a, base_rate)
    return np.diag(cost)


def best_band(cost, share, max_share=None):
    """(i, j) of the cheapest pair, optionally with review share <= max_share."""
    c = cost.copy()
    if max_share is not None:
        c = np.where(share <= max_share + 1e-12, c, np.inf)
    i, j = np.unravel_index(np.argmin(c), c.shape)
    return int(i), int(j)


def cost_do_nothing(a: Assumptions, base_rate=None):
    pi = a.base_rate if base_rate is None else base_rate
    return pi * a.cost_fn + (1 - pi) * a.cost_tn


def cost_review_everything(a: Assumptions, base_rate=None):
    pi = a.base_rate if base_rate is None else base_rate
    m, f = a.reviewer_miss, a.reviewer_false_alarm
    pos = a.cost_review + m * a.cost_fn + (1 - m) * a.cost_tp
    neg = a.cost_review + f * a.cost_fp + (1 - f) * a.cost_tn
    return pi * pos + (1 - pi) * neg


def daily_load(a: Assumptions, share: float, tpr_a, tpr_b, fpr_a, fpr_b) -> dict:
    """What an operating point means in people and alerts per day."""
    pi, cases = a.base_rate, a.cases_per_day
    reviews = share * cases
    hours = reviews * a.review_seconds / 3600
    shifts_needed = hours / (a.shift_hours * a.utilisation)
    coverage = max(shifts_needed, a.shifts_per_day)         # 24/7 needs every shift staffed
    true_alerts = pi * (tpr_b + (tpr_a - tpr_b) * (1 - a.reviewer_miss)) * cases
    false_alerts = (1 - pi) * (fpr_b + (fpr_a - fpr_b) * a.reviewer_false_alarm) * cases
    return {
        "cases_per_day": round(cases),
        "reviews_per_day": round(reviews),
        "review_share": round(share, 4),
        "reviewer_hours_per_day": round(hours, 1),
        "operator_shifts_needed_per_day": round(shifts_needed, 2),
        "operator_shifts_available_per_day": a.reviewers_per_shift * a.shifts_per_day,
        "capacity_used": round(reviews / a.review_capacity_per_day, 3),
        "people_on_roster": math.ceil(coverage * 7 / 5),      # each person works 5 of 7 days
        "true_alerts_per_day": round(true_alerts, 1),
        "false_alerts_per_day": round(false_alerts, 1),
        "smoke_cases_per_day": round(pi * cases, 1),
    }


# ------------------------------ analysis -----------------------------------
def _split(rows, positive, only_clean=True):
    rows = [r for r in rows if not (only_clean and r.get("leaky"))]
    pos = [r["scores"][positive] for r in rows if positive in r["gt_classes"]]
    neg = [r["scores"][positive] for r in rows if positive not in r["gt_classes"]]
    return np.array(pos), np.array(neg)


def _point(a, tpr, fpr, i, j, cost, share):
    return {
        "pass_below": float(THRESHOLDS[i]), "fail_at_or_above": float(THRESHOLDS[j]),
        "cost_per_1000": round(1000 * float(cost[i, j]), 1),
        "review_share": round(float(share[i, j]), 4),
        "tpr_at_fail": round(float(tpr[j]), 4), "fpr_at_fail": round(float(fpr[j]), 4),
        "load": daily_load(a, float(share[i, j]), tpr[i], tpr[j], fpr[i], fpr[j]),
    }


def analyse_split(pos, neg, a: Assumptions) -> dict:
    tpr, fpr = rates(pos, neg)
    cost, share = cost_grid(tpr, fpr, a)
    diag = np.diag(cost)
    k = int(np.argmin(diag))
    i, j = best_band(cost, share, a.review_cap_share)
    iu, ju = best_band(cost, share, None)
    d25 = int(np.where(THRESHOLDS == 0.25)[0][0])
    nothing, everything = cost_do_nothing(a), cost_review_everything(a)
    per_day = a.cases_per_day
    ev_load = daily_load(a, 1.0, np.array(1.0), np.array(0.0), np.array(1.0), np.array(0.0))
    single = _point(a, tpr, fpr, k, k, cost, share)
    band = _point(a, tpr, fpr, i, j, cost, share)
    return {
        "n_smoke_images": int(len(pos)), "n_no_smoke_images": int(len(neg)),
        "tpr": [round(float(x), 4) for x in tpr], "fpr": [round(float(x), 4) for x in fpr],
        "single_cost_per_1000": [round(1000 * float(x), 1) for x in diag],
        "single_optimum": single,
        "band_optimum": band,
        "band_optimum_unconstrained": _point(a, tpr, fpr, iu, ju, cost, share),
        "strategies": {
            "do_nothing": {"cost_per_1000": round(1000 * nothing, 1),
                           "cost_per_day": round(nothing * per_day)},
            "person_checks_everything": {"cost_per_1000": round(1000 * everything, 1),
                                         "cost_per_day": round(everything * per_day),
                                         "load": ev_load},
            "model_alone_at_0.25": {"cost_per_1000": round(1000 * float(diag[d25]), 1),
                                    "cost_per_day": round(float(diag[d25]) * per_day)},
            "model_alone_best_threshold": {"cost_per_1000": single["cost_per_1000"],
                                           "cost_per_day": round(single["cost_per_1000"] / 1000 * per_day),
                                           "threshold": single["fail_at_or_above"]},
            "model_plus_review": {"cost_per_1000": band["cost_per_1000"],
                                  "cost_per_day": round(band["cost_per_1000"] / 1000 * per_day),
                                  "pass_below": band["pass_below"],
                                  "fail_at_or_above": band["fail_at_or_above"]},
        },
    }


def bootstrap(pos, neg, a: Assumptions, n=300, seed=0) -> dict:
    """Resample the test images to see how much the chosen thresholds and costs
    would move if the test set were a different draw. Small test sets move a lot."""
    rng = np.random.default_rng(seed)
    out = {"single_t": [], "single_cost": [], "pass_below": [], "fail_at": [],
           "band_cost": [], "band_beats_single": []}
    for _ in range(n):
        p = rng.choice(pos, len(pos)); q = rng.choice(neg, len(neg))
        tpr, fpr = rates(p, q)
        cost, share = cost_grid(tpr, fpr, a)
        k = int(np.argmin(np.diag(cost)))
        i, j = best_band(cost, share, a.review_cap_share)
        out["single_t"].append(THRESHOLDS[k]); out["single_cost"].append(1000 * cost[k, k])
        out["pass_below"].append(THRESHOLDS[i]); out["fail_at"].append(THRESHOLDS[j])
        out["band_cost"].append(1000 * cost[i, j]); out["band_beats_single"].append(cost[i, j] < cost[k, k])
    pct = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 50, 95])]
    return {"resamples": n, "percentiles_5_50_95": {
        "single_threshold": pct(out["single_t"]), "single_cost_per_1000": pct(out["single_cost"]),
        "pass_below": pct(out["pass_below"]), "fail_at_or_above": pct(out["fail_at"]),
        "band_cost_per_1000": pct(out["band_cost"])},
        "share_of_resamples_where_review_band_beats_model_alone": round(float(np.mean(out["band_beats_single"])), 3)}


def sensitivity(pos, neg, a: Assumptions) -> dict:
    tpr, fpr = rates(pos, neg)
    res = {"base_rate": [], "cost_fp": [], "reviewer_miss": []}
    for pi in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1):
        cost, share = cost_grid(tpr, fpr, a, pi)
        k = int(np.argmin(np.diag(cost))); i, j = best_band(cost, share, a.review_cap_share)
        nothing, ev = cost_do_nothing(a, pi), cost_review_everything(a, pi)
        costs = {"do_nothing": nothing, "model_alone": cost[k, k], "model_plus_review": cost[i, j]}
        res["base_rate"].append({
            "base_rate": pi, "single_threshold": float(THRESHOLDS[k]),
            "band": [float(THRESHOLDS[i]), float(THRESHOLDS[j])],
            "cost_per_1000": {n: round(1000 * float(v), 1) for n, v in costs.items()},
            "cheapest": min(costs, key=costs.get)})
    # lowest base rate at which the model alone beats doing nothing
    grid = np.logspace(-6, -1, 500); breakeven = None
    for pi in grid:
        cost, _ = cost_grid(tpr, fpr, a, pi)
        if np.diag(cost).min() < cost_do_nothing(a, pi):
            breakeven = float(pi); break
    res["break_even_base_rate_model_alone_vs_do_nothing"] = breakeven
    for fp in (15.0, 50.0, 150.0, 500.0, 1500.0):
        a2 = replace(a, cost_fp=fp); cost, share = cost_grid(tpr, fpr, a2)
        k = int(np.argmin(np.diag(cost))); i, j = best_band(cost, share, a2.review_cap_share)
        res["cost_fp"].append({"cost_fp": fp, "cost_fn_over_cost_fp": round(a.cost_fn / fp, 1),
                               "single_threshold": float(THRESHOLDS[k]),
                               "band": [float(THRESHOLDS[i]), float(THRESHOLDS[j])]})
    for m in (0.0, 0.05, 0.1, 0.2, 0.4):
        a2 = replace(a, reviewer_miss=m); cost, share = cost_grid(tpr, fpr, a2)
        k = int(np.argmin(np.diag(cost))); i, j = best_band(cost, share, a2.review_cap_share)
        res["reviewer_miss"].append({"reviewer_miss": m,
                                     "model_alone_per_1000": round(1000 * float(cost[k, k]), 1),
                                     "band_per_1000": round(1000 * float(cost[i, j]), 1)})
    return res


def make_plot(result: dict, a: Assumptions, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = result["clean_test"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(THRESHOLDS, s["single_cost_per_1000"], label="model decides alone")
    ax.axhline(s["strategies"]["do_nothing"]["cost_per_1000"], ls="--", c="grey", label="do nothing")
    t = s["single_optimum"]["fail_at_or_above"]
    ax.axvline(t, c="C1", ls=":", label=f"best single threshold = {t:.2f}")
    b = s["band_optimum"]
    ax.axvspan(b["pass_below"], b["fail_at_or_above"], color="C2", alpha=0.2,
               label=f"review band {b['pass_below']:.2f}-{b['fail_at_or_above']:.2f}")
    ax.set_yscale("log")   # costs at very low thresholds are huge; log keeps the minimum visible
    ax.set_xlabel("confidence threshold (smoke)"); ax.set_ylabel("expected cost per 1,000 cases (EUR)")
    ax.set_title(f"Cost vs threshold (base rate {a.base_rate:.2%}, clean test)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def run_costing(out_dir: Path, a: Assumptions | None = None) -> dict:
    """Read test_scores.json, write cost_sweep.json, operating_point.json, cost_curve.png."""
    out_dir = Path(out_dir)
    a = a or Assumptions()
    data = json.loads((out_dir / "test_scores.json").read_text())
    positive, rows = data["positive_class"], data["images"]
    pos, neg = _split(rows, positive, only_clean=True)
    if len(pos) == 0 or len(neg) == 0:
        print("Costing skipped: the clean test set needs both smoke and no-smoke images.")
        return {}
    fpos, fneg = _split(rows, positive, only_clean=False)

    result = {
        "assumptions": asdict(a),
        "derived": {"cases_per_day": a.cases_per_day,
                    "review_capacity_per_day": round(a.review_capacity_per_day),
                    "review_cap_share": round(a.review_cap_share, 4)},
        "thresholds": [float(t) for t in THRESHOLDS],
        "positive_class": positive,
        "note_on_negatives": ("every 'no smoke' test image still contains FIRE; there are no "
                              "ordinary empty scenes, so false-alarm rates on real footage are unknown"),
        "clean_test": analyse_split(pos, neg, a),
        "full_test_leaky": analyse_split(fpos, fneg, a),
        "bootstrap_clean_test": bootstrap(pos, neg, a),
        "sensitivity_clean_test": sensitivity(pos, neg, a),
    }
    (out_dir / "cost_sweep.json").write_text(json.dumps(result, indent=1))
    band = result["clean_test"]["band_optimum"]
    (out_dir / "operating_point.json").write_text(json.dumps({
        "positive_class": positive,
        "pass_below": band["pass_below"],
        "fail_at_or_above": band["fail_at_or_above"],
        "single_threshold": result["clean_test"]["single_optimum"]["fail_at_or_above"],
        "basis": "clean test set, expected cost, review share capped by available reviewers",
    }, indent=1))
    make_plot(result, a, out_dir / "cost_curve.png")
    print_costing(result)
    return result


def print_costing(r: dict) -> None:
    c = r["clean_test"]
    print("\n=== COSTING (clean test set; per 1,000 cases, EUR) ===")
    print(f"smoke images {c['n_smoke_images']}, no-smoke images {c['n_no_smoke_images']}, "
          f"base rate {r['assumptions']['base_rate']:.3%}")
    for name, s in c["strategies"].items():
        print(f"  {name:<28} {s['cost_per_1000']:>9,.0f}   (per day {s['cost_per_day']:>9,.0f})")
    b = c["band_optimum"]
    print(f"\nchosen band: pass below {b['pass_below']:.2f}, fail at/above {b['fail_at_or_above']:.2f}")
    print("  load:", json.dumps(b["load"]))
    print("bootstrap 5/50/95%:", json.dumps(r["bootstrap_clean_test"]["percentiles_5_50_95"]))
    print("break-even base rate (model vs do nothing):",
          r["sensitivity_clean_test"]["break_even_base_rate_model_alone_vs_do_nothing"])


if __name__ == "__main__":
    run_costing(Path("artifacts"))