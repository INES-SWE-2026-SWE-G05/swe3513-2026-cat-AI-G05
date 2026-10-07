"""Tests for A3 (src/milkcheck/model.py). Do not edit; the lecturer grades with these.

These tests use their own small data, so they do not depend on A1.
"""
import numpy as np
import pytest

from milkcheck.model import fit_logistic, make_features, predict_risk, risk_label, sigmoid


def test_a3_sigmoid():
    assert sigmoid(0) == pytest.approx(0.5)
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01


def test_a3_make_features_scales_by_ten():
    X = make_features([25.0, 5.0], [2.0, 6.0])
    assert X.shape == (2, 2)
    assert X.tolist() == [[2.5, 0.2], [0.5, 0.6]]


def test_a3_fit_learns_warm_and_old_is_risky():
    temp = np.array([5, 6, 7, 8, 9, 20, 22, 25, 28, 30], dtype=float)
    hours = np.array([1, 1, 2, 1, 2, 5, 6, 6, 7, 8], dtype=float)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    w, b = fit_logistic(make_features(temp, hours), y)
    assert len(w) == 2 and isinstance(b, float)
    assert w[0] > 0 and w[1] > 0, "Warmer and older milk should raise the risk"
    assert predict_risk(w, b, 28, 7) > 0.7
    assert predict_risk(w, b, 6, 1) < 0.3


def test_a3_fit_matches_reference_numbers():
    X = make_features([5, 30, 10, 25], [1, 8, 2, 6])
    w, b = fit_logistic(X, [0, 1, 0, 1], lr=0.5, steps=3000)
    assert w == pytest.approx([5.8645, 2.2568], abs=0.01)
    assert b == pytest.approx(-10.8148, abs=0.01)


def test_a3_risk_label():
    assert [risk_label(r) for r in (0.0, 0.29, 0.3, 0.59, 0.6, 1.0)] == \
        ["Low", "Low", "Medium", "Medium", "High", "High"]
