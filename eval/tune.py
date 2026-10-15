from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results.csv"
OUTPUT = ROOT / "eval" / "tuning.json"
SIGNALS = ("ensemble_score", "ela", "fft", "noise", "metadata", "boundary")


def _metrics(rows: list[dict[str, float | str]], low: float, high: float) -> tuple[float, float, float]:
    actual = [row["expected"] == "fake" for row in rows]
    predicted = [row["score"] >= high for row in rows]
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 1.0
    return f1, fpr, (tp + tn) / len(rows)


def main() -> int:
    if not RESULTS.exists():
        print("Run eval/run_eval.py first.")
        return 1
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        source = [row for row in csv.DictReader(handle) if not row.get("error")]
    if not source or len({row["expected"] for row in source}) < 2:
        print("At least one valid real and fake result are required for tuning.")
        return 1
    rows = [
        {
            "expected": row["expected"],
            "values": {name: float(row[name]) for name in SIGNALS},
        }
        for row in source
    ]
    best: tuple[float, float, float, dict[str, float], float, float] | None = None
    for scales in itertools.product((0.0, 0.5, 1.0), repeat=5):
        weights = {"ensemble_score": 1.0}
        weights.update(dict(zip(SIGNALS[1:], scales)))
        for low in range(20, 46, 5):
            for high in range(60, 86, 5):
                if high <= low:
                    continue
                scored = [
                    {
                        "expected": row["expected"],
                        "score": 100
                        * sum(row["values"][name] * weights[name] for name in SIGNALS)
                        / sum(weights.values()),
                    }
                    for row in rows
                ]
                f1, fpr, accuracy = _metrics(scored, low, high)
                candidate = (f1, -fpr, accuracy, weights, low, high)
                if best is None or candidate[:3] > best[:3]:
                    best = candidate
    assert best is not None
    result = {
        "signal_weights": best[3],
        "real_threshold": best[4],
        "deepfake_threshold": best[5],
        "f1": best[0],
        "false_positive_rate_real": -best[1],
        "accuracy": best[2],
        "note": "Grid-search suggestion only; review on a held-out set before copying to app/config.py.",
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
