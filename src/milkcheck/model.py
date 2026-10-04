"""A3 · MEMBER 3 · Binary logistic regression for milk rejection risk

Owner (GitHub): @souleymanyayagouni
Mobile task  : M2 (DeliveryForm) in swe3409-cat1 repository

WHAT MEMBER 3 DOES
Train a tiny logistic regression model that tells the collection centre
whether a can of milk is likely to be rejected (1) or accepted (0).
No sklearn: implement sigmoid, feature extraction, and gradient descent
from scratch using only numpy.

Done means: python -m pytest tests/test_a3_model.py -v  -> 5 passed,
merged into main through a pull request reviewed by a teammate.
"""
import numpy as np
import pandas as pd


# ── Sigmoid ──────────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function: σ(z) = 1 / (1 + e^{-z}).

    Numerically stable implementation: clips z to [-500, 500].

    Parameters
    ----------
    z : np.ndarray
        Real-valued array of any shape.

    Returns
    -------
    np.ndarray
        Values in (0, 1), same shape as *z*.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ── Feature extraction ───────────────────────────────────────
def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Build the feature matrix X from a deliveries DataFrame.

    Features (in order):
        0  litres              – volume of the can
        1  temp_c              – temperature at collection
        2  hours_since_milking – freshness proxy
        3  litres * temp_c     – interaction term

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: litres, temp_c, hours_since_milking.

    Returns
    -------
    np.ndarray
        Shape (n_samples, 4), dtype float64.
    """
    X = np.column_stack([
        df["litres"].to_numpy(dtype=float),
        df["temp_c"].to_numpy(dtype=float),
        df["hours_since_milking"].to_numpy(dtype=float),
        (df["litres"] * df["temp_c"]).to_numpy(dtype=float),
    ])
    return X


# ── Logistic regression trainer ──────────────────────────────
def train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    epochs: int = 1000,
) -> np.ndarray:
    """Train binary logistic regression via gradient descent.

    Uses binary cross-entropy loss:
        L = -1/n * Σ [ y·log(p) + (1-y)·log(1-p) ]

    Update rule:
        w ← w - lr · (1/n) · Xᵀ · (p - y)

    Parameters
    ----------
    X      : np.ndarray  shape (n_samples, n_features)
    y      : np.ndarray  shape (n_samples,) — binary labels (0 or 1)
    lr     : float       learning rate (default 0.01)
    epochs : int         number of gradient-descent iterations (default 1000)

    Returns
    -------
    np.ndarray
        Weight vector w of shape (n_features + 1,) — last element is the bias.
    """
    n, d = X.shape
    # Add bias column of ones
    Xb = np.hstack([X, np.ones((n, 1))])
    w  = np.zeros(d + 1)

    for _ in range(epochs):
        p    = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n
        w   -= lr * grad

    return w


# ── Prediction helper ────────────────────────────────────────
def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return rejection probabilities for feature matrix *X*.

    Parameters
    ----------
    X : np.ndarray  shape (n_samples, n_features)
    w : np.ndarray  shape (n_features + 1,)  — from train()

    Returns
    -------
    np.ndarray  shape (n_samples,)  values in (0, 1)
    """
    n = X.shape[0]
    Xb = np.hstack([X, np.ones((n, 1))])
    return sigmoid(Xb @ w)

def predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return binary predictions (0 or 1) from probability scores."""
    return (predict_proba(X, w) >= threshold).astype(int)
