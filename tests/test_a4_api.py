"""Tests for A4 (src/milkcheck/api.py). Do not edit; the lecturer grades with these.

/summary and /risk need A1, A2 and A3 merged into main (git pull).
"""
from fastapi.testclient import TestClient

from milkcheck.api import app

client = TestClient(app)


def test_a4_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_a4_summary_has_five_sectors():
    r = client.get("/summary")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 5
    assert set(rows[0]) == {"sector", "deliveries", "total_litres", "rejection_rate"}
    assert rows[0]["sector"] == "Kinigi"


def test_a4_risk_high_and_low():
    hot = client.get("/risk", params={"temp_c": 28, "hours": 6}).json()
    cold = client.get("/risk", params={"temp_c": 6, "hours": 1}).json()
    assert set(hot) == {"temp_c", "hours", "risk", "label"}
    assert hot["label"] == "High" and hot["risk"] >= 0.6
    assert cold["label"] == "Low" and cold["risk"] < 0.3


def test_a4_risk_rejects_impossible_temperature():
    assert client.get("/risk", params={"temp_c": 99, "hours": 1}).status_code == 422


def test_a4_accepts_a_good_delivery():
    r = client.post("/deliveries", json={"farmer_id": "FRM-0012", "litres": 18.5, "temp_c": 7.0, "hours": 1.5})
    assert r.status_code == 200
    assert r.json() == {"accepted": True, "farmer_id": "FRM-0012"}


def test_a4_rejects_a_bad_delivery():
    for bad in ({"farmer_id": "frm-12", "litres": 18.5, "temp_c": 7.0, "hours": 1.5},
                {"farmer_id": "FRM-0012", "litres": 0, "temp_c": 7.0, "hours": 1.5},
                {"farmer_id": "FRM-0012", "litres": 18.5, "temp_c": 99, "hours": 1.5}):
        assert client.post("/deliveries", json=bad).status_code == 422, f"{bad} should be refused"
