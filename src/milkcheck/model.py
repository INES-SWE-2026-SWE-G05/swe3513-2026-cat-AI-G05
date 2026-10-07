"""A3 · MEMBER 3 · Learn the rejection risk (logistic regression by gradient descent)

Owner (your GitHub username): @
Your mobile task in the swe3409-cat1 repository: M2 (DeliveryForm)

WHAT MEMBER 3 DOES
Milk that arrives warm, or long after milking, is more often rejected by the
lab. You train a model that gives a risk between 0 and 1 from two inputs:
temperature (temp_c) and hours since milking. This is classification
(Day 3) trained with gradient descent (Day 2):

    risk = sigmoid(w1 * temp_c/10 + w2 * hours/10 + b)

Your tests use their own small data, so you can start at once.

Done means: python -m pytest tests/test_a3_model.py -v   -> 5 passed,
merged into main through a pull request reviewed by a teammate.
"""
import numpy as np


def sigmoid(z):
    """Return 1 / (1 + e^(-z)). z can be a number or an array.
    Tip: 1 / (1 + np.exp(-np.asarray(z, dtype=float)))"""
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return float(result) if result.ndim == 0 else result


def make_features(temp_c, hours):
    """Return a 2-column array [temp_c / 10, hours / 10], one row per delivery.

    Dividing by 10 puts both inputs on a similar scale, so gradient descent is stable.
    Steps:
    1. temp_c = np.asarray(temp_c, dtype=float).reshape(-1); same for hours
    2. return np.column_stack([temp_c / 10, hours / 10])
    """
    temp_c = np.asarray(temp_c, dtype=float).reshape(-1)
    hours = np.asarray(hours, dtype=float).reshape(-1)
    if temp_c.size != hours.size:
        raise ValueError("temp_c and hours must contain the same number of values")
    return np.column_stack([temp_c / 10, hours / 10])


def fit_logistic(X, y, lr=0.5, steps=3000):
    """Train by gradient descent. Return (w, b): w = list of 2 floats, b = float.

    Steps:
    1. X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    2. w = np.zeros(X.shape[1]); b = 0.0
    3. Repeat `steps` times:
           p   = sigmoid(X @ w + b)      # current risk for every row
           err = p - y                   # how wrong each risk is
           w  -= lr * (X.T @ err) / len(y)
           b  -= lr * np.mean(err)
    4. return [float(v) for v in w], float(b)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[0] != y.size or y.size == 0:
        raise ValueError("X and y must contain the same non-zero number of rows")
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(steps):
        probabilities = sigmoid(X @ w + b)
        error = probabilities - y
        w -= lr * (X.T @ error) / len(y)
        b -= lr * np.mean(error)
    return [float(value) for value in w], float(b)


def predict_risk(w, b, temp_c, hours):
    """Return the risk for one delivery (a float) or for many (an array).

    Steps:
    1. p = sigmoid(make_features(temp_c, hours) @ np.asarray(w) + b)
    2. return float(p[0]) if p.size == 1 else p
    """
    probabilities = sigmoid(make_features(temp_c, hours) @ np.asarray(w) + b)
    return float(probabilities[0]) if probabilities.size == 1 else probabilities


def risk_label(risk):
    """Below 0.3 -> "Low";  below 0.6 -> "Medium";  otherwise -> "High"."""
    if risk < 0.3:
        return "Low"
    if risk < 0.6:
        return "Medium"
    return "High"
