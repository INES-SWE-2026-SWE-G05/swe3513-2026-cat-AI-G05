"""A2 · MEMBER 2 · Summarise the deliveries for the manager

Owner (GitHub): @umkalsumkarim72
Mobile task  : M4 (api.ts + App.tsx) in swe3409-cat1 repository

WHAT MEMBER 2 DOES
The manager of the collection centre asks two questions every week:
"Which sector brings the most milk, and where is milk rejected most often?"
and "How much milk did we collect each day?". You answer both with groupby.
Your tests use their own small table, so you can start at once.

Done means: python -m pytest tests/test_a2_stats.py -v  -> 4 passed,
merged into main through a pull request reviewed by a teammate.
"""
import pandas as pd


def summary_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sector with these columns, in this order:

        sector | total_litres | deliveries | rejected | rejection_rate

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: sector, litres, rejected.

    Returns
    -------
    pd.DataFrame
        Sorted by total_litres descending.
        rejection_rate is a float in [0, 1] (rejected / deliveries).
    """
    grouped = (
        df.groupby("sector", as_index=False)
        .agg(
            total_litres=("litres", "sum"),
            deliveries=("litres", "count"),
            rejected=("rejected", "sum"),
        )
    )
    grouped["rejection_rate"] = grouped["rejected"] / grouped["deliveries"]
    grouped = grouped.sort_values("total_litres", ascending=False).reset_index(drop=True)
    return grouped[["sector", "total_litres", "deliveries", "rejected", "rejection_rate"]]


def litres_per_day(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per calendar day with total litres collected.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: date (datetime64[ns]), litres.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64[ns]), total_litres (float).
        Sorted by date ascending.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    daily = (
        df.groupby("date", as_index=False)
        .agg(total_litres=("litres", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily
