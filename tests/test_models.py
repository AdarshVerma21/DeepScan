from types import SimpleNamespace

from app.models import _fake_indices
from app.models import _fake_label_info


def test_fake_label_detection():
    cases = (
        ({0: "artificial", 1: "human"}, {0}),
        ({0: "Real", 1: "Fake"}, {1}),
        ({0: "real", 1: "deepfake"}, {1}),
        ({0: "AI", 1: "Human"}, {0}),
    )
    for labels, expected in cases:
        assert _fake_indices(SimpleNamespace(id2label=labels)) == expected


def test_real_labels_are_not_fake_labels():
    assert _fake_label_info(SimpleNamespace(id2label={0: "real", 1: "human"})) is None


def test_binary_unknown_label_is_the_non_real_class():
    assert _fake_label_info(
        SimpleNamespace(id2label={0: "real", 1: "class_1"})
    )[0] == {1}
