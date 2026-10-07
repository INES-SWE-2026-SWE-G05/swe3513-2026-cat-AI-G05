"""A4 · MEMBER 4 · Serve the data and the model to the phone (FastAPI)

Owner (UmkalsumKarim): @umi
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
    # TODO A4: add three fields, then delete this TODO line:
    #   litres: float, greater than 0 and at most 60   -> Field(gt=0, le=60)
    #   temp_c: float, from 0 to 45                     -> Field(ge=0, le=45)
    #   hours:  float, from 0 to 24                     -> Field(ge=0, le=24)


# TODO A4: write the function and the four endpoints below, then delete this TODO line.
#
# @lru_cache                          # train once, then reuse the result
# def trained_model():
#     df = clean_deliveries(load_deliveries())
#     X = make_features(df["temp_c"], df["hours_since_milking"])
#     return fit_logistic(X, df["rejected"])
#
# 1. GET  /health    -> {"status": "ok"}
#
# 2. GET  /summary   -> summary_by_sector(clean_deliveries(load_deliveries()))
#                       converted with .to_dict(orient="records")
#
# 3. GET  /risk?temp_c=28&hours=6
#         Parameters: temp_c: float = Query(ge=0, le=45), hours: float = Query(ge=0, le=24)
#         w, b = trained_model();  r = round(predict_risk(w, b, temp_c, hours), 2)
#         return {"temp_c": temp_c, "hours": hours, "risk": r, "label": risk_label(r)}
#
# 4. POST /deliveries  takes a Delivery d and returns {"accepted": True, "farmer_id": d.farmer_id}
#
# The pattern:
#
# @app.get("/health")
# def health():
#     return {"status": "ok"}
