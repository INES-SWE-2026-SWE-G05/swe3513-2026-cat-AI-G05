"""A4 · MEMBER 4 · FastAPI backend exposing the milk-check service

Owner (GitHub): @ibrahimhachim169
Mobile task  : M1 (logic.ts) in swe3409-cat1 repository

ENDPOINTS
---------
GET  /health      → {"status": "ok"}
GET  /summary     → sector stats + daily litres
POST /risk        → rejection probability for a single can
POST /deliveries  → store a delivery and return it with risk label

Run with:  uvicorn src.milkcheck.api:app --host 0.0.0.0 --port 8000 --reload

Done means: the four endpoints return the correct JSON shapes and the
mobile app can reach them over the local network.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .data import clean_deliveries, load_deliveries
from .model import extract_features, predict_proba, train
from .stats import litres_per_day, summary_by_sector

app = FastAPI(title="Milk Check API", version="1.0.0")

# Allow any origin so the React Native app can reach this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load and train once at startup ───────────────────────────
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "deliveries.csv"

try:
    _df   = clean_deliveries(load_deliveries(_DATA_PATH))
    _X    = extract_features(_df)
    _y    = _df["rejected"].to_numpy(float)
    _w    = train(_X, _y, lr=0.01, epochs=1000)
    _ready = True
except Exception:
    _df    = None
    _w     = None
    _ready = False


# ── Schemas ──────────────────────────────────────────────────
class RiskRequest(BaseModel):
    litres:               float = Field(gt=0, description="Volume in litres")
    temp_c:               float = Field(ge=0, le=45, description="Temperature in °C")
    hours_since_milking:  float = Field(gt=0, le=24, description="Hours since milking")


class DeliveryIn(BaseModel):
    farmer_id:            str
    litres:               float = Field(gt=0)
    temp_c:               float = Field(ge=0, le=45)
    hours_since_milking:  float = Field(gt=0, le=24)
    risk_score:           float = Field(ge=0.0, le=1.0, default=0.0)


# ── /health ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness probe – returns 200 when the server is up."""
    return {"status": "ok", "model_ready": _ready}


# ── /summary ─────────────────────────────────────────────────
@app.get("/summary")
def summary():
    """Aggregated sector statistics and daily litres totals."""
    if not _ready or _df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    sectors = summary_by_sector(_df).to_dict(orient="records")
    daily   = litres_per_day(_df)
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return {
        "sectors":      sectors,
        "litres_per_day": daily.to_dict(orient="records"),
    }


# ── /risk ─────────────────────────────────────────────────────
@app.post("/risk")
def risk(req: RiskRequest):
    """Return rejection probability for a single delivery.

    Returns
    -------
    {"risk_score": float, "risk_label": "LOW"|"MEDIUM"|"HIGH"}
    """
    if not _ready or _w is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    import pandas as pd
    row = pd.DataFrame([{
        "litres":              req.litres,
        "temp_c":              req.temp_c,
        "hours_since_milking": req.hours_since_milking,
    }])
    X_row = extract_features(row)
    score = float(predict_proba(X_row, _w)[0])

    label = "LOW" if score < 0.4 else ("MEDIUM" if score < 0.7 else "HIGH")
    return {"risk_score": round(score, 4), "risk_label": label}


# ── /deliveries ───────────────────────────────────────────────
_deliveries: list[dict] = []   # in-memory store (no DB needed for CAT1)

@app.post("/deliveries", status_code=201)
def create_delivery(delivery: DeliveryIn):
    """Record a delivery; returns it enriched with a risk label."""
    score = delivery.risk_score
    label = "LOW" if score < 0.4 else ("MEDIUM" if score < 0.7 else "HIGH")

    record = delivery.model_dump()
    record["risk_label"] = label
    _deliveries.append(record)
    return record


@app.get("/deliveries")
def list_deliveries():
    """Return all deliveries recorded in this session."""
    return _deliveries
