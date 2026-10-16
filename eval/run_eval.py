from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.forensics import run_forensics
from app.fusion import fuse
from app.image_pipeline import analyze_image
from app.video_pipeline import analyze_video

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
FORENSIC_NAMES = ("ela", "fft", "noise", "metadata", "boundary")
RESULT_FIELDS = (
    "path",
    "kind",
    "expected",
    "fake_score",
    "verdict",
    "ensemble_score",
    "per_model",
    "ela",
    "fft",
    "noise",
    "metadata",
    "boundary",
    "confidence",
    "abstained",
    "error",
)


def _files(directory: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions
    )


def _json_value(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True)


def _analyze_image_file(path: Path, expected: str) -> dict[str, Any]:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the image")
    result = analyze_image(image)
    forensic = result["forensic"]
    fusion = fuse(result["ensemble_score"], forensic, limitations=result["limitations"])
    row: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "kind": "image",
        "expected": expected,
        "fake_score": fusion["fake_score"],
        "verdict": fusion["verdict"],
        "ensemble_score": result["ensemble_score"],
        "confidence": fusion["confidence"],
        "abstained": fusion["evidence_quality"]["abstained"],
        "per_model": _json_value(result["per_model"]),
        "error": "",
    }
    for name in FORENSIC_NAMES:
        row[name] = forensic[name]["score"]
    return row


def _analyze_video_file(path: Path, expected: str) -> dict[str, Any]:
    result = analyze_video(str(path))
    frames = result["frames"]
    if not frames:
        raise ValueError("No readable video frames")
    primary = max(frames, key=lambda item: item["score"])["result"]
    forensic = run_forensics(
        primary["image"], b"", path.name, primary["face"].bbox
    )
    fusion = fuse(
        primary["ensemble_score"],
        forensic,
        result["temporal_score"],
        result["limitations"],
    )
    row: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "kind": "video",
        "expected": expected,
        "fake_score": fusion["fake_score"],
        "verdict": fusion["verdict"],
        "ensemble_score": primary["ensemble_score"],
        "confidence": fusion["confidence"],
        "abstained": fusion["evidence_quality"]["abstained"],
        "per_model": _json_value(primary["per_model"]),
        "error": "",
    }
    for name in FORENSIC_NAMES:
        row[name] = forensic[name]["score"]
    return row


def _collect(data_dir: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    failures = 0
    sources = (
        (data_dir / "real", "real"),
        (data_dir / "fake", "fake"),
    )
    for directory, expected in sources:
        for path in _files(directory, IMAGE_EXTENSIONS):
            try:
                rows.append(_analyze_image_file(path, expected))
                print(f"OK image {path.relative_to(ROOT)}")
            except Exception as exc:
                failures += 1
                print(f"ERROR image {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
                rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "kind": "image",
                        "expected": expected,
                        "error": str(exc),
                    }
                )
    video_dir = data_dir.parent / "data_video"
    for directory, expected in (
        (video_dir / "real", "real"),
        (video_dir / "fake", "fake"),
    ):
        for path in _files(directory, VIDEO_EXTENSIONS):
            try:
                rows.append(_analyze_video_file(path, expected))
                print(f"OK video {path.relative_to(ROOT)}")
            except Exception as exc:
                failures += 1
                print(f"ERROR video {path.relative_to(ROOT)}: {exc}", file=sys.stderr)
                rows.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "kind": "video",
                        "expected": expected,
                        "error": str(exc),
                    }
                )
    return rows, failures


def _binary_metrics(
    rows: list[dict[str, Any]],
    thresholds: tuple[float, float] = (30.0, 72.0),
) -> dict[str, float | list[list[int]]]:
    valid = [row for row in rows if not row.get("error")]
    if not valid:
        return {
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "roc_auc": math.nan,
            "false_positive_rate_real": math.nan,
            "false_negative_rate_fake": math.nan,
            "inconclusive_rate": math.nan,
            "expected_calibration_error": math.nan,
            "confusion_matrix": [[0, 0], [0, 0]],
        }
    actual = [row["expected"] == "fake" for row in valid]
    real_threshold, deepfake_threshold = thresholds
    # Treat the middle band as abstention, not as positive evidence.
    predicted = [
        float(row["fake_score"]) >= deepfake_threshold
        for row in valid
    ]
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(valid),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "roc_auc": _roc_auc(actual, [float(row["fake_score"]) for row in valid]),
        "false_positive_rate_real": fp / (fp + tn) if fp + tn else math.nan,
        "false_negative_rate_fake": fn / (fn + tp) if fn + tp else math.nan,
        "inconclusive_rate": sum(
            real_threshold <= float(row["fake_score"]) < deepfake_threshold for row in valid
        ) / len(valid),
        "expected_calibration_error": _ece(actual, [float(row["fake_score"]) / 100 for row in valid]),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _ece(actual: list[bool], probabilities: list[float], bins: int = 10) -> float:
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [i for i, probability in enumerate(probabilities) if lower <= probability < upper or (index == bins - 1 and probability == upper)]
        if members:
            error += len(members) / len(actual) * abs(
                sum(actual[i] for i in members) / len(members) - sum(probabilities[i] for i in members) / len(members)
            )
    return error


def _roc_auc(actual: list[bool], scores: list[float]) -> float:
    positives = sum(actual)
    negatives = len(actual) - positives
    if not positives or not negatives:
        return math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        rank = (index + end + 2) / 2
        for position in range(index, end + 1):
            ranks[order[position]] = rank
        index = end + 1
    positive_rank_sum = sum(rank for rank, is_positive in zip(ranks, actual) if is_positive)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _distribution(rows: list[dict[str, Any]], label: str, key: str) -> str:
    values = [
        float(row[key])
        for row in rows
        if row.get("expected") == label and not row.get("error") and row.get(key) != ""
    ]
    return f"mean={mean(values):.4f}, median={median(values):.4f}, n={len(values)}" if values else "n=0"


def _print_report(rows: list[dict[str, Any]]) -> None:
    metrics = _binary_metrics(rows)
    print("\nMetrics (50.0 fake-score threshold)")
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate_real", "false_negative_rate_fake", "inconclusive_rate", "expected_calibration_error"):
        value = metrics[name]
        print(f"{name}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"{name}: n/a")
    print(f"confusion_matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print("\nScore distributions")
    for key, title in (("fake_score", "final fake score"), ("ensemble_score", "ensemble")):
        print(f"{title}: real {_distribution(rows, 'real', key)}; fake {_distribution(rows, 'fake', key)}")
    model_names: set[str] = set()
    import json

    for row in rows:
        if row.get("per_model") and not row.get("error"):
            model_names.update(item["model"] for item in json.loads(row["per_model"]))
    for model_name in sorted(model_names):
        for label in ("real", "fake"):
            values = []
            for row in rows:
                if row.get("expected") != label or row.get("error"):
                    continue
                values.extend(
                    item["fake_probability"]
                    for item in json.loads(row["per_model"])
                    if item["model"] == model_name
                )
            text = f"mean={mean(values):.4f}, median={median(values):.4f}, n={len(values)}" if values else "n=0"
            print(f"model {model_name} ({label}): {text}")
    for name in FORENSIC_NAMES:
        print(
            f"forensic {name} (real): {_distribution(rows, 'real', name)}; "
            f"(fake): {_distribution(rows, 'fake', name)}"
        )


def _holdout_split(
    rows: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        path = Path(row["path"])
        stem = re.sub(r"_wa$", "", path.stem, flags=re.IGNORECASE)
        key = (row["expected"], str(path.with_name(stem + path.suffix).parent / stem))
        grouped.setdefault(key, []).append(row)
    train, holdout = [], []
    for key in sorted(grouped):
        digest = hashlib.sha256(f"42:{key[0]}:{key[1]}".encode("utf-8")).digest()
        target = holdout if int.from_bytes(digest[:8], "big") / 2**64 < fraction else train
        target.extend(grouped[key])
    return train, holdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DeepScan on labeled local media.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "eval" / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.csv")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--proposed-real-threshold", type=float, default=30.0)
    parser.add_argument("--proposed-deepfake-threshold", type=float, default=72.0)
    args = parser.parse_args()
    rows, failures = _collect(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} result rows to {args.output}")
    if not rows:
        print("No labeled files found. Add images under eval/data/real and eval/data/fake.")
    else:
        _print_report(rows)
        train, holdout = _holdout_split([row for row in rows if not row.get("error")], args.holdout_fraction)
        print(f"\nDeterministic held-out split: train={len(train)}, holdout={len(holdout)}")
        print("Current thresholds (30/72) and proposed thresholds are evaluated on the same held-out rows.")
        for label, thresholds in (
            ("current", (30.0, 72.0)),
            ("proposed", (args.proposed_real_threshold, args.proposed_deepfake_threshold)),
        ):
            metrics = _binary_metrics(holdout, thresholds)
            print(
                f"{label} held-out: fpr={metrics['false_positive_rate_real']}, "
                f"fnr={metrics['false_negative_rate_fake']}, "
                f"precision={metrics['precision']}, recall={metrics['recall']}, "
                f"inconclusive={metrics['inconclusive_rate']}"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
