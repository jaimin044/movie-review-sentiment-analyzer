"""Prediction module for Context-Aware Movie Review Sentiment Analyzer.

Provides a unified, robust inference interface for predicting sentiment from
raw movie reviews using saved serialized pipelines. Handles model loading,
caching, input validation, and confidence score calculation.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# Model registry mapping logical model names to artifact filenames
MODEL_FILES: dict[str, str] = {
    "baseline_nb": "baseline_nb.joblib",
    "baseline_logistic": "baseline_logistic.joblib",
    "context_nb": "context_nb.joblib",
    "context_logistic": "context_logistic.joblib",
}

# In-memory cache for loaded pipelines to avoid redundant disk I/O
_LOADED_PIPELINES: dict[str, Pipeline] = {}


def get_supported_models() -> list[str]:
    """Returns the list of supported model identifiers."""
    return sorted(list(MODEL_FILES.keys()))


def list_available_models(models_dir: str = "models") -> list[str]:
    """Lists model identifiers whose .joblib artifacts currently exist on disk.

    Args:
        models_dir: Directory to check for serialized model files.

    Returns:
        list[str]: Model identifiers available for inference.
    """
    available: list[str] = []
    for model_name, filename in MODEL_FILES.items():
        if os.path.exists(os.path.join(models_dir, filename)):
            available.append(model_name)
    return sorted(available)


def clear_model_cache() -> None:
    """Clears in-memory cached model pipelines."""
    global _LOADED_PIPELINES
    _LOADED_PIPELINES.clear()


def load_pipeline(
    model_name: str = "baseline_logistic",
    models_dir: str = "models",
    force_reload: bool = False,
) -> Pipeline:
    """Loads a serialized model pipeline from disk, with in-memory caching.

    Args:
        model_name: Identifier of the model to load.
        models_dir: Directory containing model artifacts.
        force_reload: If True, bypasses cache and reloads from disk.

    Returns:
        Pipeline: The loaded scikit-learn Pipeline.

    Raises:
        ValueError: If model_name is not recognized.
        FileNotFoundError: If the model artifact does not exist on disk.
        RuntimeError: If loading the model artifact fails.
    """
    if model_name not in MODEL_FILES:
        valid_models = get_supported_models()
        raise ValueError(
            f"Invalid model name '{model_name}'. Supported models are: {valid_models}"
        )

    cache_key = f"{models_dir}:{model_name}"
    if not force_reload and cache_key in _LOADED_PIPELINES:
        return _LOADED_PIPELINES[cache_key]

    filename = MODEL_FILES[model_name]
    artifact_path = os.path.join(models_dir, filename)

    if not os.path.exists(artifact_path):
        available = list_available_models(models_dir)
        raise FileNotFoundError(
            f"Model artifact for '{model_name}' not found at '{artifact_path}'. "
            f"Available trained models: {available if available else 'None'}. "
            f"Please run 'python -m src.train' to train and save baseline models."
        )

    try:
        pipeline: Pipeline = joblib.load(artifact_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load serialized model artifact from '{artifact_path}': {exc}"
        ) from exc

    _LOADED_PIPELINES[cache_key] = pipeline
    return pipeline


def validate_review_input(review: Any) -> str:
    """Validates review input text.

    Args:
        review: Raw review input to validate.

    Returns:
        str: Cleaned, non-empty review string.

    Raises:
        TypeError: If review is not a string.
        ValueError: If review is empty or whitespace-only.
    """
    if not isinstance(review, str):
        raise TypeError(
            f"Review input must be a string, got {type(review).__name__} instead."
        )

    cleaned = review.strip()
    if not cleaned:
        raise ValueError(
            "Review text cannot be empty or contain only whitespace characters."
        )

    return review


def predict_sentiment(
    review: str,
    model_name: str = "baseline_logistic",
    models_dir: str = "models",
) -> dict[str, Any]:
    """Predicts sentiment for a single raw movie review string.

    Applies the exact same preprocessing and TF-IDF transformation steps
    used during training via the loaded scikit-learn Pipeline.

    Args:
        review: Raw movie review text.
        model_name: Identifier of model to use. Defaults to 'baseline_logistic'.
        models_dir: Directory where models are stored. Defaults to 'models'.

    Returns:
        dict[str, Any]: Dictionary containing:
            - sentiment (str): 'positive' or 'negative'
            - confidence (float): Model estimated probability of predicted sentiment (0.0 to 1.0)
            - model (str): Name of the model used

    Raises:
        TypeError: If review is not a string.
        ValueError: If review is empty/whitespace or model_name is invalid.
        FileNotFoundError: If the model artifact cannot be found.

    Example:
        >>> predict_sentiment("An incredible movie with great performances!")
        {'sentiment': 'positive', 'confidence': 0.8921, 'model': 'baseline_logistic'}
    """
    valid_review = validate_review_input(review)
    pipeline = load_pipeline(model_name=model_name, models_dir=models_dir)

    # The pipeline accepts raw string in a sequence and applies TextPreprocessor -> TF-IDF -> Classifier
    predicted_label = str(pipeline.predict([valid_review])[0])

    confidence: Optional[float] = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba([valid_review])[0]
        classes = list(pipeline.classes_)
        pred_idx = classes.index(predicted_label)
        confidence = round(float(probabilities[pred_idx]), 4)
    elif hasattr(pipeline, "decision_function"):
        # Fallback for classifiers with decision function (e.g., linear SVM)
        score = float(pipeline.decision_function([valid_review])[0])
        confidence = round(float(1.0 / (1.0 + np.exp(-score))), 4)

    return {
        "sentiment": predicted_label,
        "confidence": confidence if confidence is not None else 1.0,
        "model": model_name,
    }


def predict_sentiment_batch(
    reviews: Sequence[str],
    model_name: str = "baseline_logistic",
    models_dir: str = "models",
) -> list[dict[str, Any]]:
    """Predicts sentiment for a batch of movie reviews.

    Args:
        reviews: Sequence of raw movie review strings.
        model_name: Identifier of model to use.
        models_dir: Directory where models are stored.

    Returns:
        list[dict[str, Any]]: List of prediction dictionaries.
    """
    if not reviews:
        return []

    # Validate all reviews
    valid_reviews = [validate_review_input(r) for r in reviews]
    pipeline = load_pipeline(model_name=model_name, models_dir=models_dir)

    predictions = pipeline.predict(valid_reviews)

    has_proba = hasattr(pipeline, "predict_proba")
    probabilities = pipeline.predict_proba(valid_reviews) if has_proba else None
    classes = list(pipeline.classes_) if has_proba else []

    results: list[dict[str, Any]] = []
    for idx, pred in enumerate(predictions):
        pred_label = str(pred)
        if has_proba and probabilities is not None:
            pred_idx = classes.index(pred_label)
            confidence = round(float(probabilities[idx][pred_idx]), 4)
        else:
            confidence = 1.0

        results.append({
            "sentiment": pred_label,
            "confidence": confidence,
            "model": model_name,
        })

    return results


def main() -> None:
    """CLI execution for quick sentiment prediction on a review."""
    parser = argparse.ArgumentParser(
        description="Predict sentiment of a movie review using trained pipelines."
    )
    parser.add_argument(
        "--review",
        type=str,
        required=True,
        help="Raw movie review text to analyze",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_logistic",
        choices=get_supported_models(),
        help="Model to use for prediction (default: baseline_logistic)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing saved model pipelines (default: models)",
    )
    args = parser.parse_args()

    try:
        output = predict_sentiment(
            review=args.review,
            model_name=args.model,
            models_dir=args.models_dir,
        )
        print(json.dumps(output, indent=2))
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
