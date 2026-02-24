"""ML model loading and prediction for churn risk."""

import logging
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.database import get_engine, run_sql_query

logger = logging.getLogger(__name__)

_model = None


def _load_model():
    """Load the pre-trained model from disk."""
    global _model
    if _model is not None:
        return _model

    try:
        import joblib

        settings = get_settings()
        path = Path(settings.model_path)
        if not path.exists():
            logger.warning("Model file not found: %s. Run train_model.py first.", path)
            return None

        _model = joblib.load(path)
        return _model
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
        return None


def _get_customer_features(customer_id: int) -> Optional[dict]:
    """Extract features for a customer from the database."""
    result = run_sql_query(
        f"SELECT age, region, total_spend, churn FROM customers WHERE customer_id = {customer_id}"
    )
    if result.get("error") or not result.get("results"):
        return None

    row = result["results"][0]
    return {
        "age": row.get("age", 0),
        "region": row.get("region", "unknown"),
        "total_spend": row.get("total_spend", 0.0),
    }


def _encode_features(features: dict[str, Any]) -> list[float]:
    """
    Encode customer features for model input.
    Must match training preprocessing: age, total_spend, region_encoded.
    """
    region_map = {"north": 0, "south": 1, "east": 2, "west": 3, "unknown": -1}
    region = str(features.get("region", "unknown")).lower()
    region_encoded = region_map.get(region, -1)
    if region_encoded < 0:
        region_encoded = 0

    return [
        float(features.get("age", 0)),
        float(features.get("total_spend", 0)),
        float(region_encoded),
    ]


def predict_churn(customer_id: int) -> dict[str, Any]:
    """
    Predict churn probability for a customer.
    Returns dict with customer_id, churn_probability, and risk_level.
    """
    model = _load_model()
    if model is None:
        return {
            "customer_id": customer_id,
            "churn_probability": 0.0,
            "risk_level": "unknown",
            "error": "Model not loaded. Run train_model.py first.",
        }

    features = _get_customer_features(customer_id)
    if features is None:
        return {
            "customer_id": customer_id,
            "churn_probability": 0.0,
            "risk_level": "unknown",
            "error": f"Customer {customer_id} not found.",
        }

    try:
        X = [_encode_features(features)]
        proba = model.predict_proba(X)[0][1]  # Probability of churn (class 1)

        if proba < 0.3:
            risk_level = "low"
        elif proba < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "customer_id": customer_id,
            "churn_probability": round(float(proba), 4),
            "risk_level": risk_level,
        }
    except Exception as e:
        logger.exception("Prediction failed for customer %s", customer_id)
        return {
            "customer_id": customer_id,
            "churn_probability": 0.0,
            "risk_level": "unknown",
            "error": str(e),
        }
