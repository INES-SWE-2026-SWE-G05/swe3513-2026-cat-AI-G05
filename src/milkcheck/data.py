"""A1 · MEMBER 1 · Load and clean the milk deliveries

Owner (GitHub): @parvinehuguetteissimbi
Mobile task  : M3 (DeliveryList) in swe3409-cat1 repository

WHAT MEMBER 1 DOES
Every morning the collection centre writes each can of milk into
data/deliveries.csv. The file is messy. You make the table the whole team
uses: everyone else's work starts from your two functions.

  1. load_deliveries(): read the CSV, refuse a file with missing columns,
     turn the date text into real dates.
  2. clean_deliveries(): fix the codes and remove the rows nobody can trust.

Done means: python -m pytest tests/test_a1_data.py -v (all 6 tests passing)  -> 6 passed,
merged into main through a pull request reviewed by a teammate.
"""
from pathlib import Path

import pandas as pd

__all__ = ["load_deliveries", "clean_deliveries", "count_records"]

# parents[2] goes up from src/milkcheck/data.py to the repository folder.
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "deliveries.csv"
REQUIRED_COLUMNS = ["delivery_id", "date", "sector", "farmer_id", "litres",
                    "temp_c", "hours_since_milking", "rejected"]


def load_deliveries(path=DATA_FILE) -> pd.DataFrame:
    """Read the deliveries CSV and return it as a DataFrame.

    Parameters
    ----------
    path : str or Path
        Location of the CSV file (default: data/deliveries.csv).

    Returns
    -------
    pd.DataFrame
        Columns: delivery_id (str), date (datetime64[ns]),
        sector (str), farmer_id (str), litres (float),
        temp_c (float), hours_since_milking (float), rejected (int 0/1).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    ValueError
        If any column in REQUIRED_COLUMNS is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["delivery_id"] = df["delivery_id"].astype(str).str.strip()
    df["sector"] = df["sector"].astype(str).str.strip().str.upper()
    df["farmer_id"] = df["farmer_id"].astype(str).str.strip().str.upper()

    return df


def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of *df* (never mutates the original).

    Rules (applied in order):
    1. Drop rows where ``date`` could not be parsed (NaT).
    2. Drop rows where ``litres`` is not positive (≤ 0 or NaN).
    3. Drop rows where ``temp_c`` is outside the realistic range [0, 45].
    4. Drop rows where ``hours_since_milking`` is not positive or > 24.
    5. Drop rows where ``rejected`` is not 0 or 1.
    6. Drop exact duplicate ``delivery_id`` values (keep first).

    Input (raw CSV): 68 rows
    Output (clean) : 62 rows
    """
    out = df.copy()

    # 1. invalid dates
    out = out[out["date"].notna()]

    # 2. non-positive litres
    out = out[out["litres"].notna() & (out["litres"] > 0)]

    # 3. temperature sanity
    out = out[out["temp_c"].notna() & (out["temp_c"].between(0, 45))]

    # 4. hours sanity
    out = out[
        out["hours_since_milking"].notna()
        & (out["hours_since_milking"] > 0)
        & (out["hours_since_milking"] <= 24)
    ]

    # 5. binary label
    out = out[out["rejected"].isin([0, 1])]

    # 6. duplicate IDs
    out = out.drop_duplicates(subset="delivery_id", keep="first")

    return out.reset_index(drop=True)

def count_records(path=DATA_FILE) -> dict:
    """Return {"raw": int, "clean": int} for quick smoke-testing."""
    raw = load_deliveries(path)
    clean = clean_deliveries(raw)
    return {"raw": len(raw), "clean": len(clean)}
