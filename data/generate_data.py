"""
Synthetic Nigerian PHC maternal health record generator.
Mimics DHIS2 / NHIS anonymised data schema.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

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

RISK_WEIGHTS = {
    "Lagos": 0.08, "Rivers": 0.09, "Oyo": 0.10, "Delta": 0.09,
    "Anambra": 0.10, "Enugu": 0.10, "Imo": 0.11, "Kogi": 0.15,
    "Kano": 0.14, "Kaduna": 0.16, "Borno": 0.20, "Bauchi": 0.19,
    "Niger": 0.17, "Plateau": 0.16, "Sokoto": 0.21, "Zamfara": 0.22,
    "Ebonyi": 0.18, "Kebbi": 0.20, "Jigawa": 0.19, "Taraba": 0.18,
}


def _time_to_hospital_minutes(state: str) -> float:
    base = {"South-West": 25, "South-South": 35, "South-East": 40,
            "North-Central": 55, "North-West": 70, "North-East": 90}[ZONES[state]]
    return max(5.0, RNG.normal(base, base * 0.3))


def generate_records(n: int = 50_000) -> pd.DataFrame:
    states = RNG.choice(STATES, size=n)
    base_risk = np.array([RISK_WEIGHTS[s] for s in states])

    age = RNG.integers(15, 46, size=n).astype(float)
    gravida = RNG.integers(1, 9, size=n).astype(float)
    parity = np.clip(gravida - RNG.integers(0, 3, size=n), 0, gravida).astype(float)

    systolic_bp = RNG.normal(120, 18, size=n)
    diastolic_bp = RNG.normal(80, 12, size=n)
    haemoglobin = RNG.normal(10.5, 2.1, size=n)
    bmi = RNG.normal(24.5, 4.5, size=n)
    gestational_age_weeks = RNG.integers(20, 43, size=n).astype(float)

    antenatal_visits = RNG.integers(0, 9, size=n).astype(float)
    skilled_birth_attendant = (RNG.random(n) > base_risk * 1.5).astype(int)
    prev_caesarean = (RNG.random(n) < 0.12).astype(int)
    prev_complication = (RNG.random(n) < base_risk * 0.8).astype(int)
    hiv_positive = (RNG.random(n) < 0.04).astype(int)
    diabetes = (RNG.random(n) < 0.06).astype(int)
    preeclampsia = (systolic_bp > 140) & (diastolic_bp > 90)
    severe_anaemia = haemoglobin < 7.0

    tthosp = np.array([_time_to_hospital_minutes(s) for s in states])
    rural = (tthosp > 60).astype(int)

    risk_score = (
        base_risk
        + 0.03 * (age > 35)
        + 0.02 * (gravida > 5)
        + 0.04 * (haemoglobin < 8)
        + 0.05 * preeclampsia.astype(float)
        + 0.06 * severe_anaemia.astype(float)
        + 0.03 * hiv_positive
        + 0.02 * diabetes
        + 0.04 * (antenatal_visits < 2)
        - 0.03 * skilled_birth_attendant
        + 0.02 * rural
        + 0.03 * prev_complication
    )
    risk_score = np.clip(risk_score, 0.01, 0.95)
    mortality = (RNG.random(n) < risk_score).astype(int)

    return pd.DataFrame({
        "patient_id": [f"NG-PHC-{i:06d}" for i in range(n)],
        "state": states,
        "geopolitical_zone": [ZONES[s] for s in states],
        "age": age,
        "gravida": gravida,
        "parity": parity,
        "gestational_age_weeks": gestational_age_weeks,
        "systolic_bp": systolic_bp.round(1),
        "diastolic_bp": diastolic_bp.round(1),
        "haemoglobin_gdl": haemoglobin.round(2),
        "bmi": bmi.round(1),
        "antenatal_visits": antenatal_visits,
        "skilled_birth_attendant": skilled_birth_attendant,
        "prev_caesarean": prev_caesarean,
        "prev_complication": prev_complication,
        "hiv_positive": hiv_positive,
        "diabetes": diabetes,
        "preeclampsia": preeclampsia.astype(int),
        "severe_anaemia": severe_anaemia.astype(int),
        "time_to_hospital_min": tthosp.round(1),
        "rural": rural,
        "mortality": mortality,
    })


if __name__ == "__main__":
    out = Path(__file__).parent
    df = generate_records(50_000)
    df.to_csv(out / "maternal_health.csv", index=False)
    print(f"Generated {len(df):,} records  |  mortality rate: {df.mortality.mean():.2%}")
    print(df.head(3).to_string())
