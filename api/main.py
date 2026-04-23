"""
FastAPI risk-scoring service for maternal mortality prediction.
POST /predict  →  returns risk score + top risk drivers + SMS alert recommendation.
"""

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder
import pandas as pd

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="Maternal Mortality Risk API",
    description="Real-time risk scoring for Nigerian PHC patients. "
                "Powered by XGBoost + SHAP. Integrates with DHIS2 & Africa's Talking SMS.",
    version="1.0.0",
)

_model = None
_explainer = None
_feature_names = None


def _load():
    global _model, _explainer, _feature_names
    if _model is None:
        _model = joblib.load(MODEL_DIR / "xgb_maternal.pkl")
        _feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        _explainer = shap.TreeExplainer(_model)


STATES = [
    "Lagos", "Kano", "Rivers", "Kaduna", "Oyo", "Borno", "Delta",
    "Anambra", "Bauchi", "Enugu", "Imo", "Kogi", "Niger", "Plateau",
    "Sokoto", "Zamfara", "Ebonyi", "Kebbi", "Jigawa", "Taraba",
]
ZONES = {
    "Lagos": "South-West", "Kano": "North-West", "Rivers": "South-South",
    "Kaduna": "North-West", "Oyo": "South-West", "Borno": "North-East",
    "Delta": "South-South", "Anambra": "South-East", "Bauchi": "North-East",
    "Enugu": "South-East", "Imo": "South-East", "Kogi": "North-Central",
    "Niger": "North-Central", "Plateau": "North-Central", "Sokoto": "North-West",
    "Zamfara": "North-West", "Ebonyi": "South-East", "Kebbi": "North-West",
    "Jigawa": "North-West", "Taraba": "North-East",
}
STATE_ENCODER = {s: i for i, s in enumerate(sorted(STATES))}
ZONE_ENCODER = {z: i for i, z in enumerate(sorted(set(ZONES.values())))}


class PatientRecord(BaseModel):
    state: Literal[
        "Lagos","Kano","Rivers","Kaduna","Oyo","Borno","Delta","Anambra",
        "Bauchi","Enugu","Imo","Kogi","Niger","Plateau","Sokoto","Zamfara",
        "Ebonyi","Kebbi","Jigawa","Taraba"
    ]
    age: float = Field(..., ge=15, le=55)
    gravida: int = Field(..., ge=1, le=15)
    parity: int = Field(..., ge=0, le=15)
    gestational_age_weeks: int = Field(..., ge=20, le=42)
    systolic_bp: float = Field(..., ge=60, le=220)
    diastolic_bp: float = Field(..., ge=40, le=140)
    haemoglobin_gdl: float = Field(..., ge=3.0, le=18.0)
    bmi: float = Field(..., ge=14.0, le=55.0)
    antenatal_visits: int = Field(..., ge=0, le=12)
    skilled_birth_attendant: int = Field(..., ge=0, le=1)
    prev_caesarean: int = Field(..., ge=0, le=1)
    prev_complication: int = Field(..., ge=0, le=1)
    hiv_positive: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    preeclampsia: int = Field(..., ge=0, le=1)
    severe_anaemia: int = Field(..., ge=0, le=1)
    time_to_hospital_min: float = Field(..., ge=1, le=600)
    rural: int = Field(..., ge=0, le=1)


class RiskResponse(BaseModel):
    patient_risk_score: float
    risk_category: str
    top_risk_drivers: list[dict]
    sms_alert_recommended: bool
    alert_message: str
    referral_urgency: str


@app.on_event("startup")
def startup():
    try:
        _load()
    except FileNotFoundError:
        pass


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict", response_model=RiskResponse)
def predict(record: PatientRecord):
    try:
        _load()
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run models/train.py first.")

    row = {
        "state": STATE_ENCODER.get(record.state, 0),
        "geopolitical_zone": ZONE_ENCODER.get(ZONES.get(record.state, ""), 0),
        "age": record.age,
        "gravida": record.gravida,
        "parity": record.parity,
        "gestational_age_weeks": record.gestational_age_weeks,
        "systolic_bp": record.systolic_bp,
        "diastolic_bp": record.diastolic_bp,
        "haemoglobin_gdl": record.haemoglobin_gdl,
        "bmi": record.bmi,
        "antenatal_visits": record.antenatal_visits,
        "skilled_birth_attendant": record.skilled_birth_attendant,
        "prev_caesarean": record.prev_caesarean,
        "prev_complication": record.prev_complication,
        "hiv_positive": record.hiv_positive,
        "diabetes": record.diabetes,
        "preeclampsia": record.preeclampsia,
        "severe_anaemia": record.severe_anaemia,
        "time_to_hospital_min": record.time_to_hospital_min,
        "rural": record.rural,
    }
    X = pd.DataFrame([row])[_feature_names]
    prob = float(_model.predict_proba(X)[0, 1])
    shap_vals = _explainer.shap_values(X)[0]

    drivers = sorted(
        [{"feature": f, "shap_value": round(float(v), 4)}
         for f, v in zip(_feature_names, shap_vals)],
        key=lambda x: -abs(x["shap_value"])
    )[:5]

    if prob < 0.25:
        category, urgency = "LOW", "routine"
    elif prob < 0.50:
        category, urgency = "MODERATE", "monitor_closely"
    elif prob < 0.75:
        category, urgency = "HIGH", "refer_within_24h"
    else:
        category, urgency = "CRITICAL", "emergency_transfer_now"

    alert = (
        f"MATERNAL RISK ALERT ({category}): Patient risk score {prob:.0%}. "
        f"Top factor: {drivers[0]['feature'].replace('_',' ')}. "
        f"Action: {urgency.replace('_',' ').upper()}. Contact CHW immediately."
    )

    return RiskResponse(
        patient_risk_score=round(prob, 4),
        risk_category=category,
        top_risk_drivers=drivers,
        sms_alert_recommended=prob > 0.40,
        alert_message=alert,
        referral_urgency=urgency,
    )
