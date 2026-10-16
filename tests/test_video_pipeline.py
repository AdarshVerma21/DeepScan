from app.video_pipeline import _frame_score


def test_frame_score_uses_inconclusive_fallback_without_detectors():
    assert _frame_score({"ensemble_score": None}) == 0.5


def test_frame_score_clips_detector_output():
    assert _frame_score({"ensemble_score": 1.5}) == 1.0
    assert _frame_score({"ensemble_score": -0.2}) == 0.0
