"""A2 · MEMBER 2 · Summarise the deliveries for the manager

Owner (your GitHub username): @
Your mobile task in the swe3409-cat1 repository: M4 (api.ts + App.tsx)

WHAT MEMBER 2 DOES
The manager of the collection centre asks two questions every week:
"Which sector brings the most milk, and where is milk rejected most often?"
and "How much milk did we collect each day?". You answer both with groupby.
Your tests use their own small table, so you can start at once: you do not
wait for Member 1.

Done means: python -m pytest tests/test_a2_stats.py -v   -> 4 passed,
merged into main through a pull request reviewed by a teammate.
"""
import pandas as pd


def summary_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sector with these columns, in this order:

        sector | deliveries | total_litres | rejection_rate

    - deliveries:      number of rows in the sector       ("delivery_id", "count")
    - total_litres:    sum of litres, rounded to 1 decimal ("litres", "sum")
    - rejection_rate:  mean of rejected, rounded to 2 decimals ("rejected", "mean")
    Sort by total_litres (largest first); on a tie, by sector name A-Z.
    Finish with .reset_index(drop=True).

    Tip:
        g = df.groupby("sector").agg(deliveries=("delivery_id", "count"), ...).reset_index()
        g = g.sort_values(["total_litres", "sector"], ascending=[False, True])
    """
    # TODO A2: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A2 summary_by_sector is not written yet")


def litres_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """Return columns  date | total_litres  (litres summed per day, rounded to 1 decimal),
    sorted by date, oldest first, with .reset_index(drop=True).

    Tip: df.groupby(df["date"].dt.date)["litres"].sum().round(1).reset_index()
         then rename the columns to ["date", "total_litres"].
    """
    # TODO A2: replace the line below with your code, then delete this TODO line.
    raise NotImplementedError("A2 litres_by_day is not written yet")
