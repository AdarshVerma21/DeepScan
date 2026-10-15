from app.fusion import fuse


def test_fusion_verdict_and_range():
    forensic = {name: {"score": 0.9, "finding": "suspicious"} for name in ("ela", "fft", "noise", "metadata", "boundary")}
    result = fuse(0.95, forensic)
    assert result["verdict"] == "Likely Deepfake"
    assert 0 <= result["fake_score"] <= 100


def test_fusion_is_inconclusive_for_mixed_signals():
    forensic = {name: {"score": 0.1, "finding": "normal"} for name in ("ela", "fft", "noise", "metadata", "boundary")}
    result = fuse(0.9, forensic)
    assert result["confidence"] == "Low"


def test_fusion_excludes_missing_signals_instead_of_using_neutral_evidence():
    result = fuse(0.9, {"ela": {"score": 0.9, "finding": "suspicious"}})
    assert result["fake_score"] > 70
    assert result["evidence_quality"]["available_forensic_signals"] == 1
    assert result["verdict"] == "Inconclusive"
    assert any("Insufficient" in reason for reason in result["evidence_quality"]["reasons"])


def test_fusion_abstains_on_low_quality_input():
    forensic = {
        name: {"score": 0.9, "finding": "suspicious"}
        for name in ("ela", "fft", "noise", "metadata", "boundary")
    }
    result = fuse(0.95, {**forensic, "_quality": {"low_resolution": True}})
    assert result["verdict"] == "Inconclusive"
    assert result["evidence_quality"]["abstained"] is True


def test_fusion_preserves_inconclusive_for_contradictory_signals():
    forensic = {
        "ela": {"score": 0.99, "finding": "suspicious"},
        "fft": {"score": 0.02, "finding": "normal"},
        "noise": {"score": 0.98, "finding": "suspicious"},
        "metadata": {"score": 0.01, "finding": "normal"},
        "boundary": {"score": 0.02, "finding": "normal"},
    }
    result = fuse(0.5, forensic)
    assert result["verdict"] == "Inconclusive"
    assert result["evidence_quality"]["abstained"] is True


def test_fusion_boundary_scores_keep_existing_thresholds():
    forensic = {name: {"score": 0.0, "finding": "normal"} for name in ("ela", "fft", "noise", "metadata", "boundary")}
    assert fuse(0.0, forensic)["verdict"] == "Likely Real"
    forensic = {name: {"score": 1.0, "finding": "suspicious"} for name in ("ela", "fft", "noise", "metadata", "boundary")}
    assert fuse(1.0, forensic)["verdict"] == "Likely Deepfake"


def test_unavailable_forensic_signal_is_excluded():
    forensic = {
        "noise": {"score": 1.0, "finding": "not measurable", "available": False},
        "metadata": {"score": 0.1, "finding": "normal"},
    }

    result = fuse(0.1, forensic)

    assert [item["signal"] for item in result["evidence"]] == [
        "Pretrained ensemble",
        "METADATA",
    ]
    assert result["evidence"][0]["score_kind"] == "probability"
    assert result["evidence"][1]["score_kind"] == "heuristic"
