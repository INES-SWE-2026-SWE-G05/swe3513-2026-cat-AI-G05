"""Tests for A1 (src/milkcheck/data.py). Do not edit; the lecturer grades with these."""
import pandas as pd
import pytest

from milkcheck.data import REQUIRED_COLUMNS, clean_deliveries, load_deliveries


def test_a1_reads_all_rows_and_columns():
    df = load_deliveries()
    assert len(df) == 68, "The CSV has 68 data rows (including the messy ones)"
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Column {col} is missing"


def test_a1_dates_are_dates():
    df = load_deliveries()
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "Convert date with pd.to_datetime"


def test_a1_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("delivery_id,sector\nD001,Kinigi\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_deliveries(bad)


def test_a1_clean_keeps_62_rows():
    out = clean_deliveries(load_deliveries())
    assert len(out) == 62, "6 bad rows must go: litres 0 and 120, temp 99, missing farmer, missing temp, duplicate D010"
    assert out["delivery_id"].is_unique


def test_a1_clean_fixes_codes():
    out = clean_deliveries(load_deliveries())
    assert set(out["sector"]) == {"Kinigi", "Nyange", "Busogo", "Gataraga", "Shingiro"}
    assert out["farmer_id"].str.fullmatch(r"FRM-\d{4}").all(), "Strip spaces and upper-case farmer_id"
    assert "FRM-0007" in set(out["farmer_id"]), "' frm-0007 ' should become 'FRM-0007'"


def test_a1_clean_does_not_change_input():
    df = load_deliveries()
    before = df.copy()
    clean_deliveries(df)
    pd.testing.assert_frame_equal(df, before)
