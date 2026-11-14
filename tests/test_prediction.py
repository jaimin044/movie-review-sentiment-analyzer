"""Unit tests for the prediction interface and inference pipeline."""

import os
import tempfile
import pytest

from src.predict import (
    clear_model_cache,
    get_supported_models,
    list_available_models,
    load_pipeline,
    predict_sentiment,
    predict_sentiment_batch,
    validate_review_input,
)


class TestInputValidation:
    """Tests for review input validation."""

    def test_valid_review_string(self):
        result = validate_review_input("Great movie!")
        assert result == "Great movie!"

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_review_input("")

    def test_whitespace_only_string_raises_value_error(self):
        with pytest.raises(ValueError, match="cannot be empty or contain only whitespace"):
            validate_review_input("   \n\t  ")

    def test_none_input_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            validate_review_input(None)

    def test_numeric_input_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            validate_review_input(12345)

    def test_list_input_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a string"):
            validate_review_input(["Great movie!"])


class TestModelLoadingAndValidation:
    """Tests for loading and validating model pipelines."""

    def test_get_supported_models(self):
        models = get_supported_models()
        assert "baseline_nb" in models
        assert "baseline_logistic" in models
        assert "context_nb" in models
        assert "context_logistic" in models

    def test_list_available_models(self):
        available = list_available_models(models_dir="models")
        assert "baseline_nb" in available
        assert "baseline_logistic" in available
        assert "context_nb" in available
        assert "context_logistic" in available

    def test_invalid_model_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid model name"):
            predict_sentiment("Good movie", model_name="unsupported_model_123")

    def test_missing_model_artifact_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError, match="not found"):
                predict_sentiment("Good movie", model_name="baseline_logistic", models_dir=empty_dir)

    def test_pipeline_caching_and_force_reload(self):
        clear_model_cache()
        pipe1 = load_pipeline("baseline_logistic", models_dir="models")
        pipe2 = load_pipeline("baseline_logistic", models_dir="models")
        assert pipe1 is pipe2

        pipe3 = load_pipeline("baseline_logistic", models_dir="models", force_reload=True)
        assert pipe3 is not None


ALL_MODELS = ["baseline_nb", "baseline_logistic", "context_nb", "context_logistic"]


class TestSentimentPrediction:
    """Tests for end-to-end sentiment prediction using all models."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_positive_review_prediction(self, model_name: str):
        review = "An absolute masterpiece! The acting was phenomenal, brilliant directing, and wonderful story."
        result = predict_sentiment(review, model_name=model_name)

        assert isinstance(result, dict)
        assert result["sentiment"] == "positive"
        assert result["model"] == model_name
        assert isinstance(result["confidence"], float)
        assert 0.5 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_negative_review_prediction(self, model_name: str):
        review = "A complete disaster of a film. Awful, dull, predictable, and utterly terrible acting."
        result = predict_sentiment(review, model_name=model_name)

        assert isinstance(result, dict)
        assert result["sentiment"] == "negative"
        assert result["model"] == model_name
        assert isinstance(result["confidence"], float)
        assert 0.5 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_raw_text_serialized_pipeline_reload(self, model_name: str):
        """Verifies that all 4 saved pipelines reload and predict from raw text."""
        pipe = load_pipeline(model_name=model_name, force_reload=True)
        raw_review = "The movie was not good at all, completely lacked any emotional depth or spark."
        prediction = pipe.predict([raw_review])[0]
        assert prediction in {"positive", "negative"}
        proba = pipe.predict_proba([raw_review])[0]
        assert len(proba) == 2
        assert round(float(proba.sum()), 4) == 1.0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_contrastive_review_prediction(self, model_name: str):
        """Verifies handling of contrastive clause reviews across all models."""
        review = "The acting was surprisingly good, but the screenplay was a terrible mess."
        result = predict_sentiment(review, model_name=model_name)
        assert result["sentiment"] in {"positive", "negative"}
        assert 0.5 <= result["confidence"] <= 1.0

    def test_prediction_output_structure(self):
        review = "This film was reasonably entertaining."
        result = predict_sentiment(review, model_name="baseline_logistic")

        assert set(result.keys()) == {"sentiment", "confidence", "model"}
        assert result["sentiment"] in {"positive", "negative"}
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["model"] == "baseline_logistic"

    def test_empty_review_rejection_in_predict(self):
        with pytest.raises(ValueError):
            predict_sentiment("   ")

    def test_batch_prediction(self):
        reviews = [
            "Simply incredible! One of the best movies of the year.",
            "Worst film ever made. Hated every second of it.",
        ]
        results = predict_sentiment_batch(reviews, model_name="baseline_logistic")

        assert len(results) == 2
        assert results[0]["sentiment"] == "positive"
        assert results[1]["sentiment"] == "negative"
        for item in results:
            assert set(item.keys()) == {"sentiment", "confidence", "model"}
            assert 0.0 <= item["confidence"] <= 1.0
