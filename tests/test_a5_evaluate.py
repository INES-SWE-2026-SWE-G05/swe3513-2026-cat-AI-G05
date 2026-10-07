"""Tests for A5 (src/milkcheck/evaluate.py). Do not edit; the lecturer grades with these."""
import pandas as pd

from milkcheck.evaluate import confusion, precision_recall, write_model_card


def test_a5_confusion():
    c = confusion([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_a5_precision_recall():
    assert precision_recall([1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0]) == (0.67, 0.67)
    assert precision_recall([1, 1, 1, 0], [1, 0, 0, 0]) == (1.0, 0.33)


def test_a5_no_positive_predictions_gives_zero_not_crash():
    assert precision_recall([1, 0], [0, 0]) == (0.0, 0.0)


def test_a5_model_card(tmp_path):
    df = pd.DataFrame({"rejected": [1, 1, 0, 0]})
    path = write_model_card(df, [0.9, 0.2, 0.7, 0.1], threshold=0.5, path=tmp_path / "card.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Model card: milk rejection risk")
    for needed in ("Data: 4 clean deliveries", "Threshold: 0.5", "Precision: 0.5", "Recall: 0.5",
                   "Confusion: tp=1 fp=1 fn=1 tn=1", "Limits:"):
        assert needed in text, f"The card must contain '{needed}'"
