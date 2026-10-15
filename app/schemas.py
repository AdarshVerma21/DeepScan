from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Verdict = Literal["Likely Real", "Inconclusive", "Likely Deepfake"]


class Evidence(BaseModel):
    signal: str
    score: float = Field(ge=0, le=1)
    finding: str
    weight: float = Field(ge=0, le=1)


class ModelOutput(BaseModel):
    model: str
    fake_probability: float = Field(ge=0, le=1)
    label: str = ""


class AnalysisResponse(BaseModel):
    id: str
    type: Literal["image", "video"]
    fake_score: float = Field(ge=0, le=100)
    verdict: Verdict
    confidence: Literal["Low", "Medium", "High"]
    evidence: list[Evidence]
    evidence_quality: dict[str, Any] = {}
    per_model: list[ModelOutput]
    timeline: list[dict[str, float | int]] = []
    heatmaps: list[str] = []
    summary_markdown: str
    processing_time: float
    limitations: list[str] = []
    metadata: dict[str, Any] = {}
    original_image: str = ""
    reason_chips: list[dict[str, str]] = []
    no_face_detected: bool = False
    input_warnings: list[str] = []
    captions: dict[str, str] = {}
    visualizations: dict[str, str] = {}
    summary_is_fallback: bool = False
    top_suspicious_frames: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
