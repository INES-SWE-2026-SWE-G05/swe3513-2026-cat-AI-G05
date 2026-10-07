"""Tests for A2 (src/milkcheck/stats.py). Do not edit; the lecturer grades with these.

These tests build their own small table, so they do not depend on A1.
"""
import pandas as pd

from milkcheck.stats import litres_by_day, summary_by_sector


def small():
    return pd.DataFrame({
        "delivery_id": ["D1", "D2", "D3", "D4", "D5"],
        "date": pd.to_datetime(["2026-10-01", "2026-10-01", "2026-10-02", "2026-10-02", "2026-10-03"]),
        "sector": ["Kinigi", "Kinigi", "Nyange", "Busogo", "Busogo"],
        "litres": [10.0, 20.5, 40.0, 5.0, 5.0],
        "rejected": [0, 1, 0, 1, 1],
    })


def test_a2_summary_columns_and_order():
    s = summary_by_sector(small())
    assert list(s.columns) == ["sector", "deliveries", "total_litres", "rejection_rate"]
    assert list(s["sector"]) == ["Nyange", "Kinigi", "Busogo"], "Sort by total_litres, largest first"


def test_a2_summary_values():
    s = summary_by_sector(small()).set_index("sector")
    assert s.loc["Kinigi", "deliveries"] == 2
    assert s.loc["Kinigi", "total_litres"] == 30.5
    assert s.loc["Kinigi", "rejection_rate"] == 0.5
    assert s.loc["Busogo", "rejection_rate"] == 1.0


def test_a2_summary_tie_sorted_by_name():
    df = small()
    df.loc[2, "litres"] = 30.5          # Nyange now ties with Kinigi
    s = summary_by_sector(df)
    assert list(s["sector"])[:2] == ["Kinigi", "Nyange"], "On a tie, sort by sector name A-Z"


def test_a2_litres_by_day():
    d = litres_by_day(small())
    assert list(d.columns) == ["date", "total_litres"]
    assert list(d["total_litres"]) == [30.5, 45.0, 5.0]
