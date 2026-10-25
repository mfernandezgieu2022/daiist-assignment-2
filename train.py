"""
Assignment 2: training pipeline.

What this does, start to finish (no manual steps):
  1. Download the pinned Roboflow dataset and describe it from its labels.
  2. Check for near-duplicate images between train/valid and test (leakage).
  3. Finetune YOLO11 from the pretrained COCO checkpoint; save weights/best.pt.
  4. Evaluate on the test split: overall + per-class + confusion matrix, on the
     full test set AND on a "clean" test set with leaked images removed.
  5. Compare against two baselines: pretrained COCO zero-shot, and trivial
     (predict nothing / always alert).
  6. Save the per-image test scores that the cost sweep will use.

Run with: uv run python main.py train

Optional environment variables:
  SMOKE_TEST=1  tiny fast run that only checks the code works. It writes to
                weights_smoke/ and artifacts_smoke/ and is NOT your real model.
  EPOCHS, MODEL, DEVICE   override the defaults below
  SKIP_TRAIN=1  reuse weights/best.pt instead of training again
"""
import hashlib
import json
import os
import platform
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from api_key import load_api_key

# --- Pinned dataset (record these exact values in REPORT.md) ---------------
WORKSPACE = "sayed-gamall"
PROJECT = "fire-smoke-detection-yolov11"
VERSION = 2

# --- Settings (record where you trained and these values in REPORT.md) -----
SMOKE_TEST = os.getenv("SMOKE_TEST") == "1"
MODEL = os.getenv("MODEL", "yolo11s.pt")
EPOCHS = int(os.getenv("EPOCHS", "1" if SMOKE_TEST else "50"))
IMGSZ = 320 if SMOKE_TEST else 640
BATCH = 8 if SMOKE_TEST else 16
FRACTION = 0.02 if SMOKE_TEST else 1.0      # share of train images used
PATIENCE = 15                                # early stopping on val
DEVICE = os.getenv("DEVICE") or None         # None = GPU if present, else CPU
SEED = 0
POSITIVE_CLASS = "smoke"                     # the class the decision is about
HASH_DISTANCE = 5                            # dHash bits: <= this = near-duplicate
EVAL_CONF = 0.25                             # threshold for the confusion matrix

DATA_DIR = Path("data/fire-smoke")           # downloaded data: do NOT commit
ARTIFACTS = Path("artifacts_smoke" if SMOKE_TEST else "artifacts")   # commit
WEIGHTS = Path("weights_smoke" if SMOKE_TEST else "weights")         # commit
RUNS = Path("runs")                          # ultralytics scratch: do NOT commit
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# =========================== 1. Dataset ====================================
def download_dataset(api_key: str) -> Path:
    """Download the pinned dataset version (skipped if already on disk)."""
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already in {DATA_DIR}, skipping download.")
        return DATA_DIR
    from roboflow import Roboflow

    project = Roboflow(api_key=api_key).workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov11", location=str(DATA_DIR))
    return Path(dataset.location)


def _images(root: Path, split: str) -> list[Path]:
    return sorted(
        p for p in (root / split / "images").iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )


def _label_rows(root: Path, split: str, img: Path) -> list[list[str]]:
    f = root / split / "labels" / f"{img.stem}.txt"
    if not f.exists():
        return []
    return [r.split() for r in f.read_text().splitlines() if r.strip()]


def _box_area(row: list[str]) -> float:
    """Box area as a fraction of the image. Handles box rows and polygon rows."""
    if len(row) == 5:
        return float(row[3]) * float(row[4])
    xs, ys = [float(v) for v in row[1::2]], [float(v) for v in row[2::2]]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _pct(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(int(q * len(sorted_values)), len(sorted_values) - 1)]


def describe_dataset(root: Path) -> dict:
    """Instances per class, split sizes, object size, background images."""
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    stats = {"classes": names, "splits": {}}
    sources = {}
    hash_to_splits = defaultdict(set)

    for split in SPLITS:
        images = _images(root, split)
        instances, images_with = Counter(), Counter()
        areas = defaultdict(list)
        background = 0

        for img in images:
            rows = _label_rows(root, split, img)
            if not rows:
                background += 1
            for r in rows:
                instances[int(r[0])] += 1
                areas[int(r[0])].append(_box_area(r))
            for c in {int(r[0]) for r in rows}:
                images_with[c] += 1
            hash_to_splits[hashlib.md5(img.read_bytes()).hexdigest()].add(split)

        # Roboflow names augmented copies "<original>.rf.<hash>.jpg"
        sources[split] = {p.name.split(".rf.")[0] for p in images}

        per_class = {}
        for c, name in enumerate(names):
            a = sorted(areas[c])
            per_class[name] = {
                "instances": instances[c],
                "images_with_class": images_with[c],
                "box_area_median": round(_pct(a, 0.5), 5) if a else None,
                "box_area_p10": round(_pct(a, 0.1), 5) if a else None,
                "box_area_p90": round(_pct(a, 0.9), 5) if a else None,
            }
        stats["splits"][split] = {
            "images": len(images),
            "unique_source_images": len(sources[split]),
            "background_images": background,
            "per_class": per_class,
        }

    stats["leakage_by_filename_and_md5"] = {
        "exact_duplicate_images_across_splits": sum(
            len(s) > 1 for s in hash_to_splits.values()
        ),
        "shared_source_names_train_test": len(sources["train"] & sources["test"]),
        "shared_source_names_train_valid": len(sources["train"] & sources["valid"]),
        "shared_source_names_valid_test": len(sources["valid"] & sources["test"]),
    }
    return stats


# =========================== 2. Leakage check ==============================
def _bits_to_int(a: np.ndarray) -> int:
    bits = (a[:, 1:] > a[:, :-1]).flatten()          # 8x8 = 64 bits
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def _hash_images(paths: list[Path]):
    """dHash of each image and of its mirror image (catches flip augmentation)."""
    h, h_flip = [], []
    for p in paths:
        with Image.open(p) as im:
            small = im.convert("L").resize((9, 8), Image.LANCZOS)
        a = np.asarray(small, dtype=np.int16)
        h.append(_bits_to_int(a))
        h_flip.append(_bits_to_int(a[:, ::-1]))
    return np.array(h, dtype=np.uint64), np.array(h_flip, dtype=np.uint64)


def _popcount(x: np.ndarray) -> np.ndarray:
    u = np.uint64
    x = x - ((x >> u(1)) & u(0x5555555555555555))
    x = (x & u(0x3333333333333333)) + ((x >> u(2)) & u(0x3333333333333333))
    x = (x + (x >> u(4))) & u(0x0F0F0F0F0F0F0F0F)
    return ((x * u(0x0101010101010101)) >> u(56)).astype(np.int64)


def _min_distance(query, ref, ref_flip) -> np.ndarray:
    """For each query hash, the smallest Hamming distance to any reference image."""
    out = np.empty(len(query), dtype=np.int64)
    for i in range(0, len(query), 64):
        q = query[i:i + 64, None]
        d = np.minimum(_popcount(q ^ ref[None, :]), _popcount(q ^ ref_flip[None, :]))
        out[i:i + 64] = d.min(axis=1)
    return out


def check_leakage(root: Path) -> tuple[dict, set, dict]:
    """Find near-duplicate images across splits by looking at the pixels.

    This is a lower bound: crops and rotations are not caught, flips are.
    """
    imgs = {s: _images(root, s) for s in SPLITS}
    h = {s: _hash_images(imgs[s]) for s in SPLITS}

    def nearest(query: str, refs: list[str]) -> np.ndarray:
        ref = np.concatenate([h[s][0] for s in refs])
        ref_flip = np.concatenate([h[s][1] for s in refs])
        return _min_distance(h[query][0], ref, ref_flip)

    d_valid_train = nearest("valid", ["train"])
    d_test_train = nearest("test", ["train"])
    d_test_any = nearest("test", ["train", "valid"])

    leaky = {imgs["test"][i].name for i in np.where(d_test_any <= HASH_DISTANCE)[0]}
    distance = {p.name: int(d) for p, d in zip(imgs["test"], d_test_any)}

    # Do images that share a filename with train really look the same?
    train_sources = {p.name.split(".rf.")[0] for p in imgs["train"]}
    shared = [i for i, p in enumerate(imgs["test"])
              if p.name.split(".rf.")[0] in train_sources]
    confirmed = sum(d_test_any[i] <= HASH_DISTANCE for i in shared)

    summary = {
        "method": f"dHash 8x8 with mirror, near-duplicate if distance <= {HASH_DISTANCE}",
        "test_images": len(imgs["test"]),
        "test_near_duplicate_of_train": int((d_test_train <= HASH_DISTANCE).sum()),
        "test_near_duplicate_of_train_or_valid": len(leaky),
        "test_share_leaky": round(len(leaky) / len(imgs["test"]), 4),
        "test_near_duplicates_by_max_distance": {
            str(t): int((d_test_any <= t).sum()) for t in (0, 2, 5, 8, 12)
        },
        "valid_images": len(imgs["valid"]),
        "valid_near_duplicate_of_train": int((d_valid_train <= HASH_DISTANCE).sum()),
        "test_images_sharing_filename_with_train": len(shared),
        "of_those_confirmed_by_pixels": int(confirmed),
        "leaky_test_images": sorted(leaky),
    }
    return summary, leaky, distance


def prepare_eval_data(root: Path, names: list[str], leaky: set) -> dict:
    """Write absolute-path yaml files, plus a clean test set without leaked images."""
    clean = root / "test_clean"
    if clean.exists():
        shutil.rmtree(clean)
    (clean / "images").mkdir(parents=True)
    (clean / "labels").mkdir(parents=True)
    for img in _images(root, "test"):
        if img.name in leaky:
            continue
        shutil.copy2(img, clean / "images" / img.name)
        label = root / "test" / "labels" / f"{img.stem}.txt"
        if label.exists():
            shutil.copy2(label, clean / "labels" / label.name)

    base = {"path": str(root.resolve()), "names": dict(enumerate(names)),
            "train": "train/images", "val": "valid/images"}
    paths = {}
    for key, test_dir in [("full", "test/images"), ("clean", "test_clean/images")]:
        path = root / f"data_{key}.yaml"
        path.write_text(yaml.safe_dump({**base, "test": test_dir}))
        paths[key] = path
    return paths


# =========================== 3. Training ==================================
def train_model(data_yaml: Path) -> Path:
    """Finetune from the pretrained checkpoint and copy best weights to weights/."""
    from ultralytics import YOLO

    WEIGHTS.mkdir(exist_ok=True)
    target = WEIGHTS / "best.pt"
    if os.getenv("SKIP_TRAIN") == "1" and target.exists():
        print(f"SKIP_TRAIN=1: reusing {target}")
        return target

    model = YOLO(MODEL)
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        fraction=FRACTION, patience=PATIENCE, seed=SEED, device=DEVICE,
        project=str(RUNS.resolve()), name="finetune", exist_ok=True, plots=True,
    )
    shutil.copy2(Path(model.trainer.best), target)

    ARTIFACTS.mkdir(exist_ok=True)
    for f in ("results.csv", "results.png"):
        src = Path(model.trainer.save_dir) / f
        if src.exists():
            shutil.copy2(src, ARTIFACTS / f"training_{f}")

    import torch
    settings = {
        "base_model": MODEL, "epochs_requested": EPOCHS, "imgsz": IMGSZ,
        "batch": BATCH, "fraction": FRACTION, "patience": PATIENCE, "seed": SEED,
        "dataset": f"{WORKSPACE}/{PROJECT}/v{VERSION}",
        "trained_on": {
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "smoke_test": SMOKE_TEST,
    }
    (ARTIFACTS / "training_settings.json").write_text(json.dumps(settings, indent=2))
    return target


# =========================== 4. Evaluation ================================
def _val(model, data_yaml: Path, conf: float, plots: bool = False):
    # ultralytics only fills the confusion matrix when plots=True
    return model.val(
        data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH, conf=conf,
        iou=0.6, device=DEVICE, plots=plots, verbose=False,
        project=str(RUNS.resolve()), name="val", exist_ok=True,
    )


def _summarize_val(m, names: list[str]) -> dict:
    """Overall and per-class detection metrics. P and R are at the F1-best threshold."""
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": round(float(m.box.p[i]), 4),
            "recall": round(float(m.box.r[i]), 4),
            "ap50": round(float(m.box.ap50[i]), 4),
            "map50_95": round(float(m.box.ap[i]), 4),
        }
    return {
        "overall": {
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
            "ap50": round(float(m.box.map50), 4), "map50_95": round(float(m.box.map), 4),
        },
        "per_class": per_class,
    }


def image_scores(model, images: list[Path], root: Path, names: list[str],
                 leaky: set, leak_distance: dict) -> list[dict]:
    """Per test image: which classes are really there, and the model's best
    confidence for each class. This is all the cost sweep needs."""
    rows = []
    results = model.predict(
        source=[str(p) for p in images], conf=0.001, iou=0.6, imgsz=IMGSZ,
        max_det=100, stream=True, verbose=False, device=DEVICE,
    )
    for img, r in zip(images, results):
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        scores = {n: round(float(conf[cls == c].max()), 4) if (cls == c).any() else 0.0
                  for c, n in enumerate(names)}
        gt = sorted({int(x[0]) for x in _label_rows(root, "test", img)})
        rows.append({"image": img.name, "gt_classes": [names[c] for c in gt],
                     "scores": scores, "leaky": img.name in leaky,
                     "leak_distance": leak_distance[img.name]})
    return rows


def image_level(rows: list[dict], positive: str, threshold: float) -> dict:
    """Image-level confusion for 'is there a <positive> in this image?'.
    Alert if the model's best <positive> confidence is >= threshold."""
    tp = fp = fn = tn = 0
    for r in rows:
        actual = positive in r["gt_classes"]
        alert = r["scores"][positive] >= threshold
        tp += actual and alert
        fp += (not actual) and alert
        fn += actual and (not alert)
        tn += (not actual) and (not alert)
    return {
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
    }


def evaluate_split(model, data_yaml: Path, rows: list[dict], names: list[str],
                   positive: str) -> dict:
    m_ap = _val(model, data_yaml, conf=0.001)          # for AP / mAP
    m_cm = _val(model, data_yaml, conf=EVAL_CONF, plots=True)   # confusion matrix
    return {
        "n_images": len(rows),
        "detection": _summarize_val(m_ap, names),
        # rows = predicted class, columns = true class, last index = background
        "confusion_matrix": {
            "conf": EVAL_CONF, "labels": names + ["background"],
            "matrix": np.asarray(m_cm.confusion_matrix.matrix).astype(int).tolist(),
        },
        "image_level": image_level(rows, positive, EVAL_CONF),
    }


# =========================== 5. Baselines =================================
def coco_zero_shot(images: list[Path], names: list[str]) -> dict:
    """The untouched pretrained COCO model. COCO has no fire or smoke class, so it
    cannot detect either by construction; we record what it does detect."""
    from ultralytics import YOLO

    coco = YOLO(MODEL)
    counts = Counter()
    for r in coco.predict(source=[str(p) for p in images], conf=EVAL_CONF, imgsz=IMGSZ,
                          stream=True, verbose=False, device=DEVICE):
        for c in r.boxes.cls.int().tolist():
            counts[coco.names[c]] += 1
    wanted = {n.lower() for n in names}
    matching = [n for n in coco.names.values() if n.lower() in wanted]
    return {
        "coco_classes_matching_target_names": matching,
        "detections_by_coco_class_top10": counts.most_common(10),
        "result": ("ap50 = 0 and recall = 0 by construction: no COCO class maps to "
                   "the target classes" if not matching else "needs mapping"),
    }


# =========================== main =========================================
def main() -> None:
    api_key = load_api_key()
    root = download_dataset(api_key)
    ARTIFACTS.mkdir(exist_ok=True)

    stats = describe_dataset(root)
    (ARTIFACTS / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    names = stats["classes"]
    positive = next(n for n in names if n.lower() == POSITIVE_CLASS)

    print("Checking for near-duplicate images between splits ...")
    leak_summary, leaky, leak_distance = check_leakage(root)
    (ARTIFACTS / "leakage.json").write_text(json.dumps(leak_summary, indent=2))
    print({k: v for k, v in leak_summary.items() if k != "leaky_test_images"})

    yamls = prepare_eval_data(root, names, leaky)
    weights = train_model(yamls["full"])

    from ultralytics import YOLO
    model = YOLO(str(weights))

    test_images = _images(root, "test")
    rows = image_scores(model, test_images, root, names, leaky, leak_distance)
    clean_rows = [r for r in rows if not r["leaky"]]
    (ARTIFACTS / "test_scores.json").write_text(json.dumps(
        {"positive_class": positive, "classes": names, "images": rows}))

    metrics = {
        "model": MODEL, "weights": str(weights), "positive_class": positive,
        "full_test": evaluate_split(model, yamls["full"], rows, names, positive),
        "clean_test": evaluate_split(model, yamls["clean"], clean_rows, names, positive),
        "baselines": {
            "coco_zero_shot": coco_zero_shot(test_images, names),
            "predict_nothing": image_level(rows, positive, threshold=1.01),
            "always_alert": image_level(rows, positive, threshold=0.0),
        },
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n=== RESULTS (also saved in artifacts/) ===")
    for split in ("full_test", "clean_test"):
        d = metrics[split]
        print(f"\n{split} ({d['n_images']} images)")
        print("  overall:", d["detection"]["overall"])
        for n, v in d["detection"]["per_class"].items():
            print(f"  {n}: {v}")
    print("\nbaselines:", json.dumps(metrics["baselines"], indent=2))
    # NEXT STAGE: cost sweep, review band, chosen operating point.


if __name__ == "__main__":
    main()