from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier   # FIXED: was DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from model_loader import save_artifacts

LOGGER = logging.getLogger(__name__)

# 📁 Paths
MODULE_DIR = Path(__file__).resolve().parent
DATA_PATH = MODULE_DIR / "performance_data.csv"


def train_models(test_size: float = 0.2, random_state: int = 42) -> Dict:
    """
    Train:
    - Performance Model
    - Promotion Model
    - Skill Gap Model
    """

    # =========================
    # Load Dataset
    # =========================
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # =========================
    # Features & Targets
    # =========================
    feature_cols = [
        "attendance",
        "task_completion_rate",
        "peer_reviews",
        "project_success_rate",
    ]

    required_cols = feature_cols + ["performance", "promotion", "skill_gap"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in dataset: {col}")

    X = df[feature_cols]
    y_perf = df["performance"]
    y_promo = df["promotion"]
    y_skill = df["skill_gap"]

    # =========================
    # Single Split
    # =========================
    X_train, X_test, y_perf_train, y_perf_test, y_promo_train, y_promo_test, y_skill_train, y_skill_test = train_test_split(
        X,
        y_perf,
        y_promo,
        y_skill,
        test_size=test_size,
        random_state=random_state,
        stratify=y_perf,   # FIXED: stratify to preserve class balance
    )

    # =========================
    # Models — RandomForest (more robust than DecisionTree)
    # FIXED: DecisionTree was overfitting on noisy synthetic data
    # =========================
    perf_model  = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    promo_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)
    skill_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_state)

    # =========================
    # Training
    # =========================
    perf_model.fit(X_train, y_perf_train)
    promo_model.fit(X_train, y_promo_train)
    skill_model.fit(X_train, y_skill_train)

    # =========================
    # Evaluation
    # =========================
    perf_acc  = accuracy_score(y_perf_test,  perf_model.predict(X_test))
    promo_acc = accuracy_score(y_promo_test, promo_model.predict(X_test))
    skill_acc = accuracy_score(y_skill_test, skill_model.predict(X_test))

    print("\n📊 Performance Model Report:")
    print(classification_report(y_perf_test, perf_model.predict(X_test)))

    print("📊 Promotion Model Report:")
    print(classification_report(y_promo_test, promo_model.predict(X_test)))

    print("📊 Skill Gap Model Report:")
    print(classification_report(y_skill_test, skill_model.predict(X_test)))

    # =========================
    # Metadata
    # =========================
    metadata = {
        "model_name": "RandomForestClassifier",   # FIXED
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "features": feature_cols,
        "dataset_rows": len(df),
        "metrics": {
            "performance_accuracy": round(perf_acc, 4),
            "promotion_accuracy":   round(promo_acc, 4),
            "skill_gap_accuracy":   round(skill_acc, 4),
        },
    }

    # =========================
    # Save Models
    # =========================
    save_artifacts(
        perf_model,
        promo_model,
        skill_model,
        metadata,
    )

    LOGGER.info("✅ Models trained and saved successfully")

    print("\n🚀 Training Completed")
    print(metadata)

    return metadata


if __name__ == "__main__":
    train_models()
