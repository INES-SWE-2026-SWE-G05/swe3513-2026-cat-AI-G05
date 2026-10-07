"""A5 · MEMBER 5 · Check the model honestly and write its model card

Owner (your GitHub username): @
Your mobile task in the swe3409-cat1 repository: M5 (health.ts + StatusBanner)

WHAT MEMBER 5 DOES
A risk score is only useful if we know how often it is right. You count the
four kinds of result (confusion matrix), compute precision and recall, and
write docs/model_card.md, the one page a manager reads before trusting the
model. Your tests use their own small data, so you can start at once.

Groups of 4 have no Member 5: this file stays as it is and nobody is marked on it.

Done means: python -m pytest tests/test_a5_evaluate.py -v   -> 4 passed,
merged into main through a pull request reviewed by a teammate.
Bonus for the demo: create docs/model_card.md from the real data and push it.
"""
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd


def confusion(y_true, y_pred):
    """Return {"tp": .., "fp": .., "fn": .., "tn": ..} as ints (1 = rejected).

    tp: truly rejected and predicted rejected      fp: truly accepted but predicted rejected
    fn: truly rejected but predicted accepted       tn: truly accepted and predicted accepted
    Tip: t = np.asarray(y_true).astype(int); p = ...; tp = int(((t == 1) & (p == 1)).sum())
    """
    truth = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return {
        "tp": int(((truth == 1) & (predicted == 1)).sum()),
        "fp": int(((truth == 0) & (predicted == 1)).sum()),
        "fn": int(((truth == 1) & (predicted == 0)).sum()),
        "tn": int(((truth == 0) & (predicted == 0)).sum()),
    }


def precision_recall(y_true, y_pred):
    """Return (precision, recall), each rounded to 2 decimals.

    precision = tp / (tp + fp)   - of the cans we flagged, how many were really rejected
    recall    = tp / (tp + fn)   - of the cans really rejected, how many we flagged
    If a denominator is 0, use 0.0 instead of crashing.
    """
    counts = confusion(y_true, y_pred)
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    return round(precision, 2), round(recall, 2)


def write_model_card(df: pd.DataFrame, risks, threshold=0.5, path="docs/model_card.md") -> Path:
    """Write the model card and return its Path. The lines, in this order:

        # Model card: milk rejection risk
        (empty line)
        Purpose: warn the collector before the lab test. A person still decides.
        Data: <len(df)> clean deliveries (synthetic, for teaching).
        Inputs: temp_c, hours_since_milking.
        Threshold: <threshold>
        Precision: <precision>
        Recall: <recall>
        Confusion: tp=<tp> fp=<fp> fn=<fn> tn=<tn>
        Limits: small synthetic data; not checked on real milk.

    y_pred = (np.asarray(risks) >= threshold).astype(int); compare it with df["rejected"].
    Tips: path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
          path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
    """
    predictions = (np.asarray(risks) >= threshold).astype(int)
    counts = confusion(df["rejected"], predictions)
    precision, recall = precision_recall(df["rejected"], predictions)
    lines = [
        "# Model card: milk rejection risk",
        "",
        "Purpose: warn the collector before the lab test. A person still decides.",
        f"Data: {len(df)} clean deliveries (synthetic, for teaching).",
        "Inputs: temp_c, hours_since_milking.",
        f"Threshold: {threshold}",
        f"Precision: {precision}",
        f"Recall: {recall}",
        (
            f"Confusion: tp={counts['tp']} fp={counts['fp']} "
            f"fn={counts['fn']} tn={counts['tn']}"
        ),
        "Limits: small synthetic data; not checked on real milk.",
    ]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
