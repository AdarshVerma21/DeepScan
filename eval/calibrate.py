from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results.csv"
OUTPUT = ROOT / "eval" / "calibration.json"


def _fit_platt(scores: Iterable[float], labels: Iterable[int]) -> tuple[float, float]:
    x = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    x = np.log(np.clip(x, 1e-6, 1 - 1e-6) / np.clip(1 - x, 1e-6, 1))
    a, b = 1.0, 0.0
    for _ in range(500):
        z = np.clip(a * x + b, -30, 30)
        probability = 1 / (1 + np.exp(-z))
        gradient_a = float(np.mean((probability - y) * x))
        gradient_b = float(np.mean(probability - y))
        a -= 0.05 * gradient_a
        b -= 0.05 * gradient_b
    return a, b


def main() -> int:
    if not RESULTS.exists():
        print(f"{RESULTS} does not exist; run eval/run_eval.py first.")
        return 1
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if not row.get("error")]
    if not rows:
        print("No valid evaluation rows; add labeled files before calibration.")
        return 1
    # Fit only on the deterministic training partition. run_eval.py reports
    # metrics on the complementary held-out partition.
    training_rows = [
        row
        for row in rows
        if int.from_bytes(hashlib.sha256(row["path"].encode("utf-8")).digest()[:8], "big") / 2**64
        >= 0.2
    ]
    if not training_rows:
        print("No training rows remain after the deterministic 80/20 split.")
        return 1
    print(f"Using {len(training_rows)} training rows; held-out rows are excluded from calibration.")
    model_values: dict[str, list[tuple[float, int]]] = {}
    for row in training_rows:
        for item in json.loads(row["per_model"]):
            model_values.setdefault(item["model"], []).append(
                (float(item.get("raw_fake_probability", item["fake_probability"])), int(row["expected"] == "fake"))
            )
    calibration = {}
    for name, values in model_values.items():
        if len({label for _, label in values}) < 2:
            print(f"Skipping {name}: both real and fake labels are required.")
            continue
        a, b = _fit_platt((score for score, _ in values), (label for _, label in values))
        calibration[name] = {"a": round(a, 8), "b": round(b, 8)}
        print(f"{name}: a={a:.6f}, b={b:.6f}, n={len(values)}")
    OUTPUT.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {OUTPUT}")
    print("Calibration is probability mapping only; detector weights and parameters are not trained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
