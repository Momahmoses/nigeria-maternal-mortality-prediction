# Maternal Mortality Prediction & Intervention Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

XGBoost + SHAP maternal mortality risk scoring for Nigerian Primary Health Centres — DHIS2-compatible REST API with SMS alerts for community health workers to intervene before emergencies occur.

---

## Problem Statement

Nigeria accounts for ~20% of global maternal deaths — most are preventable with timely intervention. Community health workers lack a systematic risk screening tool for antenatal visits. This engine scores every pregnant woman's risk at registration and triggers alerts for high-risk cases.

---

## Features

| Feature | Description |
|---------|-------------|
| XGBoost Risk Scoring | Probability of adverse maternal outcome per patient |
| SHAP Explainability | Per-patient risk factor breakdown for health workers |
| DHIS2 Integration | REST API compatible with Nigeria's national health data system |
| SMS Alerts | Automatic Twilio SMS to CHWs for high-risk cases |
| Geospatial Risk Map | State-level maternal mortality risk dashboard |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | XGBoost, SHAP |
| API | FastAPI, Uvicorn |
| Alerts | Twilio SMS |
| Geospatial | GeoPandas, Folium |
| Data | pandas, NumPy |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/nigeria-maternal-mortality-prediction.git
cd nigeria-maternal-mortality-prediction
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Data Sources

- NDHS (Nigeria Demographic and Health Survey)
- DHIS2 Nigeria PHC antenatal records
- WHO maternal health indicators
- GRID3 health facility geolocation data

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
