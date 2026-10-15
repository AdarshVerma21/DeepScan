from __future__ import annotations

import numpy as np

from .config import (
    DEEPFAKE_THRESHOLD,
    LOW_QUALITY_SIGNAL_MULTIPLIER,
    REAL_THRESHOLD,
    SIGNAL_WEIGHTS,
)


def fuse(
    ensemble_score: float | None,
    forensic: dict[str, dict[str, object]],
    temporal_score: float | None = None,
    limitations: list[str] | None = None,
) -> dict[str, object]:
    evidence: list[dict[str, object]] = []
    if ensemble_score is not None and np.isfinite(float(ensemble_score)):
        evidence.append({"signal": "Pretrained ensemble", "score": float(np.clip(ensemble_score, 0, 1)), "finding": "Average calibrated fake probability across available detectors.", "weight": SIGNAL_WEIGHTS["ensemble"], "score_kind": "probability"})
    quality = forensic.get("_quality", {})
    low_quality = bool(
        quality.get("low_resolution")
        or quality.get("heavy_compression")
        or quality.get("screenshot_like")
    )
    available_forensic: list[tuple[str, dict[str, object]]] = []
    for name in ("ela", "fft", "noise", "metadata", "boundary"):
        item = forensic.get(name)
        if (
            not isinstance(item, dict)
            or item.get("available", True) is False
            or not np.isfinite(float(item.get("score", np.nan)))
        ):
            continue
        available_forensic.append((name, item))
    for name, item in available_forensic:
        weight = SIGNAL_WEIGHTS[name]
        # ELA, FFT, noise and boundary all inspect related image artifacts.
        # Damp their combined influence instead of counting correlated cues independently.
        if name in {"ela", "fft", "noise", "boundary"}:
            weight *= 0.5
        if low_quality and name in {"ela", "fft", "noise"}:
            weight *= LOW_QUALITY_SIGNAL_MULTIPLIER
        evidence.append({"signal": name.upper(), "score": float(np.clip(item["score"], 0, 1)), "finding": str(item.get("finding", "")), "weight": weight, "score_kind": "heuristic"})
    if low_quality:
        evidence.append({"signal": "LIMITATION", "score": 0.5, "finding": str(quality.get("finding", "Input quality limits forensic reliability.")), "weight": 0.0})
    total_weight = sum(float(item["weight"]) for item in evidence)
    score = 50.0 if not total_weight else 100 * sum(float(item["score"]) * float(item["weight"]) for item in evidence) / total_weight
    if temporal_score is not None and np.isfinite(float(temporal_score)):
        score = 100 * ((score / 100) * (1 - 0.2) + float(np.clip(temporal_score, 0, 1)) * 0.2)
    scores = [float(item["score"]) for item in evidence if float(item["weight"]) > 0]
    spread = float(np.std(scores)) if scores else 1.0
    quality_reasons: list[str] = []
    if quality.get("low_resolution"):
        quality_reasons.append("Low-resolution input limits detector and forensic reliability.")
    if quality.get("heavy_compression"):
        quality_reasons.append("Heavy compression limits forensic reliability.")
    if ensemble_score is None or not any(item["signal"] == "Pretrained ensemble" for item in evidence):
        quality_reasons.append("No pretrained detector output was available.")
    if len(available_forensic) < 2:
        quality_reasons.append("Insufficient independent forensic signals were available.")
    contradictory = spread > 0.28
    if contradictory:
        quality_reasons.append("Signals materially disagree, so the result is abstained rather than over-interpreted.")
    abstained = bool(quality_reasons or limitations) and (
        bool(quality.get("low_resolution") or quality.get("heavy_compression"))
        or contradictory
        or ensemble_score is None
        or len(evidence) < 3
    )
    verdict = "Inconclusive" if abstained else ("Likely Real" if score < REAL_THRESHOLD else ("Likely Deepfake" if score >= DEEPFAKE_THRESHOLD else "Inconclusive"))
    confidence = "Low" if spread > 0.28 or limitations or quality_reasons else ("High" if spread < 0.12 else "Medium")
    return {
        "fake_score": round(float(np.clip(score, 0, 100)), 2),
        "verdict": verdict,
        "confidence": confidence,
        "evidence": evidence,
        "evidence_quality": {
            "abstained": abstained,
            "reasons": list(dict.fromkeys(quality_reasons + list(limitations or []))),
            "available_forensic_signals": len(available_forensic),
            "signal_spread": round(spread, 4),
            "score_semantics": "A heuristic fake-likelihood index, not a calibrated probability.",
        },
    }
