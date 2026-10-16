from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results.csv"
OUTPUT = ROOT / "eval" / "tuning.json"
SIGNALS = ("ensemble_score", "ela", "fft", "noise", "metadata", "boundary")


def _metrics(rows: list[dict[str, float | str]], low: float, high: float) -> tuple[float, float, float, float]:
    actual = [row["expected"] == "fake" for row in rows]
    predicted = [float(row["score"]) >= high for row in rows]
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 1.0
    inconclusive = sum(low <= float(row["score"]) < high for row in rows) / len(rows)
    return f1, fpr, (tp + tn) / len(rows), inconclusive


def _split(rows: list[dict[str, float | str]], holdout_fraction: float) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    grouped: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for row in rows:
        path = Path(str(row["path"]))
        stem = re.sub(r"_wa$", "", path.stem, flags=re.IGNORECASE)
        key = (str(row["expected"]), str(path.with_name(stem + path.suffix).parent / stem))
        grouped.setdefault(key, []).append(row)
    train, holdout = [], []
    for key in sorted(grouped):
        digest = hashlib.sha256(f"42:{key[0]}:{key[1]}".encode("utf-8")).digest()
        target = holdout if int.from_bytes(digest[:8], "big") / 2**64 < holdout_fraction else train
        target.extend(grouped[key])
    return train, holdout


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tune fusion on the 70% training partition only.")
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    args = parser.parse_args()
    if not RESULTS.exists():
        print("Run eval/run_eval.py first.")
        return 1
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        source = [row for row in csv.DictReader(handle) if not row.get("error")]
    if not source or len({row["expected"] for row in source}) < 2:
        print("At least one valid real and fake result are required for tuning.")
        return 1
    rows = [
        {"path": row["path"], "expected": row["expected"], "values": {name: float(row[name]) for name in SIGNALS}}
        for row in source
    ]
    train, holdout = _split(rows, args.holdout_fraction)
    if not train or not holdout:
        print("The grouped 70/30 split did not produce both training and held-out rows.")
        return 1
    best: tuple[float, float, float, float, dict[str, float], float, float] | None = None
    for scales in itertools.product((0.0, 0.5, 1.0), repeat=5):
        weights = {"ensemble_score": 1.0}
        weights.update(dict(zip(SIGNALS[1:], scales)))
        for low in range(20, 46, 5):
            for high in range(60, 86, 5):
                if high <= low:
                    continue
                if not sum(weights.values()):
                    continue
                scored = [
                    {
                        "expected": row["expected"],
                        "score": 100 * sum(row["values"][name] * weights[name] for name in SIGNALS) / sum(weights.values()),
                    }
                    for row in train
                ]
                f1, fpr, accuracy, inconclusive = _metrics(scored, low, high)
                if fpr > 0.05:
                    continue
                candidate = (inconclusive, f1, -fpr, accuracy, weights, low, high)
                if best is None or candidate[:4] > best[:4]:
                    best = candidate
    if best is None:
        print("No candidate met the required real-image false-positive rate <= 5%.")
        return 1
    result = {
        "signal_weights": best[4],
        "real_threshold": best[5],
        "deepfake_threshold": best[6],
        "training_inconclusive_rate": best[0],
        "training_f1": best[1],
        "training_false_positive_rate_real": -best[2],
        "training_accuracy": best[3],
        "train_rows": len(train),
        "held_out_rows": len(holdout),
        "note": "Tuned only on the training partition. Evaluate the held-out partition separately before promotion.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
