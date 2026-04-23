"""
Train XGBoost maternal mortality risk model with SHAP explainability.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "maternal_health.csv"
MODEL_DIR = Path(__file__).parent
CATEGORICAL = ["state", "geopolitical_zone"]
DROP_COLS = ["patient_id", "mortality"]


def load_features(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS)
    y = df["mortality"]
    for col in CATEGORICAL:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X.astype(float), y


def train():
    df = pd.read_csv(DATA_PATH)
    X, y = load_features(df)

    pos_weight = (y == 0).sum() / (y == 1).sum()
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"\nTest ROC-AUC: {roc:.4f}  |  Avg Precision: {ap:.4f}")
    print(classification_report(y_test, (y_prob > 0.4).astype(int)))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.iloc[:500])
    feature_importance = dict(zip(
        X.columns.tolist(),
        np.abs(shap_values).mean(axis=0).tolist()
    ))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: -x[1]))

    joblib.dump(model, MODEL_DIR / "xgb_maternal.pkl")
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")

    metrics = {"roc_auc": round(roc, 4), "avg_precision": round(ap, 4)}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODEL_DIR / "shap_importance.json").write_text(
        json.dumps(feature_importance, indent=2)
    )
    print("\nTop 5 SHAP features:")
    for k, v in list(feature_importance.items())[:5]:
        print(f"  {k}: {v:.4f}")

    return model


if __name__ == "__main__":
    train()
