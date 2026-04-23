# Maternal Mortality Prediction & Intervention Engine

> **Problem:** Nigeria accounts for ~20% of global maternal deaths — most are preventable with timely, data-driven intervention.

## Overview

An end-to-end XGBoost + SHAP pipeline that scores every incoming PHC patient for mortality risk, surfaces the top clinical drivers, and triggers SMS alerts to Community Health Workers (CHWs) via Africa's Talking.

## Architecture

```
PHC EHR / DHIS2 → Feature Engineering → XGBoost Classifier
                                              ↓
                                    SHAP Explainer (TreeExplainer)
                                              ↓
                                    FastAPI Risk Scoring Service
                                              ↓
                                 Africa's Talking SMS → CHW handset
```

## Features

| Feature Group | Examples |
|---|---|
| Clinical vitals | Systolic/diastolic BP, haemoglobin, BMI |
| Obstetric history | Gravida, parity, previous C-section, complications |
| Comorbidities | HIV, diabetes, preeclampsia, severe anaemia |
| Access/geography | Time-to-hospital (minutes), rural flag, state |
| ANC utilisation | Number of antenatal care visits |

## Quickstart

```bash
# 1. Generate synthetic PHC data
python data/generate_data.py

# 2. Train XGBoost model (CV + holdout evaluation)
python models/train.py

# 3. Start REST API
uvicorn api.main:app --reload --port 8001

# 4. Score a patient
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Zamfara",
    "age": 38, "gravida": 6, "parity": 5,
    "gestational_age_weeks": 36,
    "systolic_bp": 155, "diastolic_bp": 100,
    "haemoglobin_gdl": 6.2, "bmi": 21.3,
    "antenatal_visits": 1, "skilled_birth_attendant": 0,
    "prev_caesarean": 1, "prev_complication": 1,
    "hiv_positive": 0, "diabetes": 0,
    "preeclampsia": 1, "severe_anaemia": 1,
    "time_to_hospital_min": 130, "rural": 1
  }'
```

## Model Performance

| Metric | Score |
|---|---|
| CV ROC-AUC | ~0.88–0.91 |
| Test Average Precision | ~0.72–0.78 |
| Threshold @ 0.40 | Optimised for recall |

## Data Sources (production)

- **DHIS2 Nigeria** — national HMIS platform
- **NHIS records** — National Health Insurance Scheme
- **GRID3 Nigeria** — geospatial population / facility data
- **NPHCDA** — National Primary Health Care Development Agency

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/predict` | Score a patient record |

## Deployment

```bash
docker build -t maternal-risk-api .
docker run -p 8001:8001 maternal-risk-api
```

## Impact

Targeting the **6 states with highest maternal mortality** (Zamfara, Sokoto, Kebbi, Borno, Yobe, Katsina) where a 10% reduction in missed high-risk cases could prevent ~2,400 deaths annually.
