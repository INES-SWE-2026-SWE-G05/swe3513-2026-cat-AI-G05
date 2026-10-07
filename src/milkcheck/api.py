"""A4 · MEMBER 4 · Serve the data and the model to the phone (FastAPI)

Owner (your GitHub username): @
Your mobile task in the swe3409-cat1 repository: M1 (logic.ts)

WHAT MEMBER 4 DOES
You are the AI integrator. The phone app (swe3409-cat1 repository) calls
your API to check the server, to get the risk of a delivery and to send it.
You USE the work of Members 1, 2 and 3: run  git pull  after they merge.
Start with /health and /deliveries: they need nobody else.

Run the server:       uvicorn milkcheck.api:app --reload --app-dir src
For the phones:       uvicorn milkcheck.api:app --host 0.0.0.0 --app-dir src
Then open:            http://127.0.0.1:8000/docs

Done means: python -m pytest tests/test_a4_api.py -v   -> 6 passed,
merged into main through a pull request reviewed by a teammate.
"""
from functools import lru_cache  # noqa: F401

from fastapi import FastAPI, Query  # noqa: F401
from pydantic import BaseModel, Field

# These imports work once Members 1, 2 and 3 have merged (git pull).
from milkcheck.data import clean_deliveries, load_deliveries  # noqa: F401
from milkcheck.model import fit_logistic, make_features, predict_risk, risk_label  # noqa: F401
from milkcheck.stats import summary_by_sector  # noqa: F401

app = FastAPI(title="Milk Check API")


class Delivery(BaseModel):
    """What the phone sends. Pydantic refuses anything else with error 422."""
    farmer_id: str = Field(pattern=r"^FRM-\d{4}$")
    litres: float = Field(gt=0, le=60)
    temp_c: float = Field(ge=0, le=45)
    hours: float = Field(ge=0, le=24)


@lru_cache
def trained_model():
    """Train the model once and reuse it for subsequent risk requests."""
    df = clean_deliveries(load_deliveries())
    features = make_features(df["temp_c"], df["hours_since_milking"])
    return fit_logistic(features, df["rejected"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/summary")
async def summary():
    df = clean_deliveries(load_deliveries())
    return summary_by_sector(df).to_dict(orient="records")


@app.get("/risk")
async def risk(
    temp_c: float = Query(ge=0, le=45),
    hours: float = Query(ge=0, le=24),
):
    weights, bias = trained_model()
    probability = round(predict_risk(weights, bias, temp_c, hours), 2)
    return {
        "temp_c": temp_c,
        "hours": hours,
        "risk": probability,
        "label": risk_label(probability),
    }


@app.post("/deliveries")
async def create_delivery(delivery: Delivery):
    return {"accepted": True, "farmer_id": delivery.farmer_id}
