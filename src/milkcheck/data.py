"""A1 · MEMBER 1 · Load and clean the milk deliveries

Owner (your GitHub username): @
Your mobile task in the swe3409-cat1 repository: M3 (DeliveryList)

WHAT MEMBER 1 DOES
Every morning the collection centre writes each can of milk into
data/deliveries.csv. The file is messy. You make the table the whole team
uses: everyone else's work starts from your two functions.

  1. load_deliveries(): read the CSV, refuse a file with missing columns,
     turn the date text into real dates.
  2. clean_deliveries(): fix the codes and remove the rows nobody can trust.

Done means: python -m pytest tests/test_a1_data.py -v   -> 6 passed,
merged into main through a pull request reviewed by a teammate.
"""
from pathlib import Path

import pandas as pd

# parents[2] goes up from src/milkcheck/data.py to the repository folder.
DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "deliveries.csv"
REQUIRED_COLUMNS = ["delivery_id", "date", "sector", "farmer_id", "litres",
                    "temp_c", "hours_since_milking", "rejected"]


def load_deliveries(path=DATA_FILE) -> pd.DataFrame:
    """Read the deliveries CSV and return it as a DataFrame.

    Steps:
    1. df = pd.read_csv(path)
    2. missing = the REQUIRED_COLUMNS that are not in df.columns.
       If missing is not empty: raise ValueError(f"Missing columns: {missing}")
    3. df["date"] = pd.to_datetime(df["date"])
    4. return df
    """
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df


def clean_deliveries(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned COPY of df (never change the original).

    Steps, in this order:
    1. out = df.copy()
    2. farmer_id: remove spaces, upper case      " frm-0007 " -> "FRM-0007"
       Tip: out["farmer_id"].astype("string").str.strip().str.upper()
    3. sector: remove spaces, Title case         " kinigi "   -> "Kinigi"   (.str.title())
    4. Drop rows where farmer_id is missing or "".
    5. Keep litres greater than 0 and at most 60:   out["litres"].gt(0) & out["litres"].le(60)
    6. Keep temp_c between 0 and 45:                out["temp_c"].between(0, 45)
       (between also drops missing temperatures)
    7. Drop duplicate delivery_id rows, keep the first one.
    8. out["rejected"] = out["rejected"].astype(int)
    9. return out.reset_index(drop=True)
    On the real file: 68 rows in, 62 rows out.
    """
    out = df.copy()
    out["farmer_id"] = out["farmer_id"].astype("string").str.strip().str.upper()
    out["sector"] = out["sector"].astype("string").str.strip().str.title()
    out = out[out["farmer_id"].notna() & out["farmer_id"].ne("")]
    out = out[out["litres"].gt(0) & out["litres"].le(60)]
    out = out[out["temp_c"].between(0, 45)]
    out = out.drop_duplicates(subset="delivery_id", keep="first")
    out["rejected"] = out["rejected"].astype(int)
    return out.reset_index(drop=True)
