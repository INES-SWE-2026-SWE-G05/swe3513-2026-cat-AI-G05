"""A5 · MEMBER 5 · Evaluate the logistic regression model

Owner (GitHub): @gueylo
Mobile task  : M5 (health.ts + StatusBanner) in swe3409-cat1 repository

WHAT MEMBER 5 DOES
Measure how well Member 3's model actually works on held-out data.
Implement confusion matrix, precision, recall, F1, and write a tidy
model card to docs/model_card.md that the whole team can share.

Done means: python -m pytest tests/test_a5_evaluate.py -v  -> 5 passed,
merged into main through a pull request reviewed by a teammate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["confusion_matrix", "precision", "recall", "f1_score", "accuracy", "write_model_card"]


# ── Confusion matrix ─────────────────────────────────────────
def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """Compute TP, FP, TN, FN for binary labels.

    Parameters
    ----------
    y_true : np.ndarray  shape (n,)  — ground-truth labels (0 or 1)
    y_pred : np.ndarray  shape (n,)  — predicted labels    (0 or 1)

    Returns
    -------
    dict with keys: "TP", "FP", "TN", "FN"
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}


# ── Precision ────────────────────────────────────────────────
def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP).  Returns 0.0 when (TP + FP) == 0."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom if denom else 0.0


# ── Recall ───────────────────────────────────────────────────
def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN).  Returns 0.0 when (TP + FN) == 0."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom if denom else 0.0


# ── F1 score ─────────────────────────────────────────────────
def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean of precision and recall.  Returns 0.0 when both are 0."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


# ── Model card generator ─────────────────────────────────────
_CARD_TEMPLATE = """\
# Model Card – Milk Rejection Risk Classifier

## Overview
Binary logistic regression trained to predict whether a can of raw milk
will be rejected at the collection centre.

| Property      | Value                        |
|---------------|------------------------------|
| Task          | Binary classification        |
| Algorithm     | Logistic regression (numpy)  |
| Training set  | 62 cleaned delivery records  |
| Features      | litres, temp_c, hours_since_milking, litres×temp_c |
| Label         | rejected (0 = accepted, 1 = rejected) |

## Performance on held-out split (80/20)

| Metric    | Value   |
|-----------|---------|
| Precision | {precision:.3f} |
| Recall    | {recall:.3f}    |
| F1 score  | {f1:.3f}        |

### Confusion matrix

|                 | Predicted 0 | Predicted 1 |
|-----------------|-------------|-------------|
| **Actual 0**    | {TN}        | {FP}        |
| **Actual 1**    | {FN}        | {TP}        |

## Intended use
Assist field collectors in flagging high-risk deliveries for on-site
temperature re-check before transport. **Not** for final rejection decisions.

## Limitations
* Trained on a small dataset (62 rows). Performance may degrade in sectors
  with unusual seasonal patterns.
* Model weights are not persisted between server restarts.

## Authors
* A5: {author_a5} (evaluation)
* A3: Souleyman Yayagouni (model)
* A1: Parvine Huguette Issimbi (data cleaning)
"""


def write_model_card(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path | None = None,
    author_a5: str = "Gueylo",
) -> str:
    """Render and write the model card markdown.

    Parameters
    ----------
    y_true    : ground-truth labels
    y_pred    : predicted labels (threshold applied by caller)
    out_path  : where to write the .md file (default: docs/model_card.md)
    author_a5 : name to embed in the card

    Returns
    -------
    str  The rendered markdown string.
    """
    if out_path is None:
        out_path = Path(__file__).resolve().parents[2] / "docs" / "model_card.md"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    card = _CARD_TEMPLATE.format(
        precision=precision(y_true, y_pred),
        recall=recall(y_true, y_pred),
        f1=f1_score(y_true, y_pred),
        author_a5=author_a5,
        **cm,
    )
    out_path.write_text(card, encoding="utf-8")
    return card

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(np.mean(y_true == y_pred))

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a formatted string report."""
    lines = [
        f"Precision : {precision(y_true, y_pred):.3f}",
        f"Recall    : {recall(y_true, y_pred):.3f}",
        f"F1 score  : {f1_score(y_true, y_pred):.3f}",
        f"Accuracy  : {accuracy(y_true, y_pred):.3f}",
    ]
    return "\n".join(lines)
