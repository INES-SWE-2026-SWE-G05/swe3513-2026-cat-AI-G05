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
    # TODO A3: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A3 sigmoid is not written yet")


def make_features(temp_c, hours):
    """Return a 2-column array [temp_c / 10, hours / 10], one row per delivery.

    Dividing by 10 puts both inputs on a similar scale, so gradient descent is stable.
    Steps:
    1. temp_c = np.asarray(temp_c, dtype=float).reshape(-1); same for hours
    2. return np.column_stack([temp_c / 10, hours / 10])
    """
    # TODO A3: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A3 make_features is not written yet")


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
    # TODO A3: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A3 fit_logistic is not written yet")


def predict_risk(w, b, temp_c, hours):
    """Return the risk for one delivery (a float) or for many (an array).

    Steps:
    1. p = sigmoid(make_features(temp_c, hours) @ np.asarray(w) + b)
    2. return float(p[0]) if p.size == 1 else p
    """
    # TODO A3: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A3 predict_risk is not written yet")


def risk_label(risk):
    """Below 0.3 -> "Low";  below 0.6 -> "Medium";  otherwise -> "High"."""
    # TODO A3: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A3 risk_label is not written yet")
