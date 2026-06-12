import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR

CANONICAL_ARTIFACT_DIR = PROJECT_ROOT / "ml_model" / "models" / "performance_prediction_models"
LEGACY_ARTIFACT_DIR = PROJECT_ROOT / "ml_model" / "models" / "performance_prediction"

def _has_required_artifacts(path: Path) -> bool:
    return (
        (path / "performance_model.pkl").exists()
        and (path / "promotion_model.pkl").exists()
        and (path / "skill_model.pkl").exists()
    )

def _resolve_artifact_dir() -> Path:
    env_override = os.getenv("PERFORMANCE_MODEL_DIR")
    if env_override:
        return Path(env_override).resolve()
    for path in [CANONICAL_ARTIFACT_DIR, LEGACY_ARTIFACT_DIR]:
        if _has_required_artifacts(path):
            return path.resolve()
    return CANONICAL_ARTIFACT_DIR.resolve()

ARTIFACT_DIR = _resolve_artifact_dir()

PERFORMANCE_MODEL_PATH = ARTIFACT_DIR / "performance_model.pkl"
PROMOTION_MODEL_PATH = ARTIFACT_DIR / "promotion_model.pkl"
SKILL_MODEL_PATH = ARTIFACT_DIR / "skill_model.pkl"
METADATA_PATH = ARTIFACT_DIR / "performance_metadata.json"

_performance_model = None
_promotion_model = None
_skill_model = None
_metadata = None

def _load_pickle(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found at {path}")
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading {name}: {e}") from e

def _write_pickle(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def reset_cache():
    global _performance_model, _promotion_model, _skill_model, _metadata
    _performance_model = None
    _promotion_model = None
    _skill_model = None
    _metadata = None

def load_performance_model():
    global _performance_model
    if _performance_model is None:
        _performance_model = _load_pickle(PERFORMANCE_MODEL_PATH, "Performance Model")
    return _performance_model

def load_promotion_model():
    global _promotion_model
    if _promotion_model is None:
        _promotion_model = _load_pickle(PROMOTION_MODEL_PATH, "Promotion Model")
    return _promotion_model

def load_skill_model():
    global _skill_model
    if _skill_model is None:
        _skill_model = _load_pickle(SKILL_MODEL_PATH, "Skill Model")
    return _skill_model

def load_metadata() -> Dict[str, Any]:
    global _metadata
    if _metadata is not None:
        return _metadata
    if not METADATA_PATH.exists():
        _metadata = {}
        return _metadata
    try:
        _metadata = json.loads(METADATA_PATH.read_text())
        return _metadata
    except Exception as e:
        raise RuntimeError(f"Error loading metadata: {e}") from e

def save_artifacts(perf_model, prom_model, skill_model, metadata: Dict[str, Any]):
    _write_pickle(PERFORMANCE_MODEL_PATH, perf_model)
    _write_pickle(PROMOTION_MODEL_PATH, prom_model)
    _write_pickle(SKILL_MODEL_PATH, skill_model)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    reset_cache()