from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from model_loader import (
    load_performance_model,
    load_promotion_model,
    load_skill_model,
)

LOGGER = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "attendance",
    "task_completion_rate",
    "peer_reviews",
    "project_success_rate",
]


class PerformancePredictionError(RuntimeError):
    """Raised when prediction fails."""


def _to_float(value: Any, field: str) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception as e:
        raise ValueError(f"Invalid value for {field}: {value}") from e


def _preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert input into model-ready DataFrame with strict validation.
    
    Expected input ranges:
      attendance          → int/float, 0–100  (percentage)
      task_completion_rate → float, 0.0–1.0  (decimal)
      peer_reviews         → float, 1.0–5.0  (rating)
      project_success_rate → float, 0.0–1.0  (decimal)
    """
    attendance           = _to_float(data.get("attendance"),           "attendance")
    task_completion_rate = _to_float(data.get("task_completion_rate"), "task_completion_rate")
    peer_reviews         = _to_float(data.get("peer_reviews"),         "peer_reviews")
    project_success_rate = _to_float(data.get("project_success_rate"), "project_success_rate")

    # =========================
    # FIXED: Input validation — catches scale mismatches early
    # =========================
    if not (0 <= attendance <= 100):
        raise ValueError(f"attendance must be between 0 and 100, got {attendance}")
    if not (0.0 <= task_completion_rate <= 1.0):
        raise ValueError(f"task_completion_rate must be between 0.0 and 1.0, got {task_completion_rate}. "
                         f"Hint: pass 0.92 not 92.")
    if not (1.0 <= peer_reviews <= 5.0):
        raise ValueError(f"peer_reviews must be between 1.0 and 5.0, got {peer_reviews}")
    if not (0.0 <= project_success_rate <= 1.0):
        raise ValueError(f"project_success_rate must be between 0.0 and 1.0, got {project_success_rate}. "
                         f"Hint: pass 0.90 not 90.")

    processed = {
        "attendance":           attendance,
        "task_completion_rate": task_completion_rate,
        "peer_reviews":         peer_reviews,
        "project_success_rate": project_success_rate,
    }

    return pd.DataFrame([processed])[FEATURE_COLUMNS]


def predict_employee_performance(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict employee performance, promotion readiness, and skill gap.
    """
    try:
        features    = _preprocess_input(data)

        perf_model  = load_performance_model()
        prom_model  = load_promotion_model()
        skill_model = load_skill_model()

        performance = perf_model.predict(features)[0]
        promotion   = prom_model.predict(features)[0]
        skill_gap   = skill_model.predict(features)[0]

        return {
            "status": "success",
            "data": {
                "performance":    str(performance),
                "promotion_ready": str(promotion),
                "skill_gap":       str(skill_gap),
            }
        }

    except Exception as error:
        LOGGER.error(f"Prediction failed: {error}")
        raise PerformancePredictionError(f"Prediction failed: {error}") from error
