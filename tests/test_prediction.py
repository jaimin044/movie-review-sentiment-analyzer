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

    def test_list_available_models(self):
        available = list_available_models(models_dir="models")
        assert "baseline_nb" in available
        assert "baseline_logistic" in available

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


class TestSentimentPrediction:
    """Tests for end-to-end sentiment prediction using baseline models."""

    @pytest.mark.parametrize("model_name", ["baseline_nb", "baseline_logistic"])
    def test_positive_review_prediction(self, model_name: str):
        review = "An absolute masterpiece! The acting was phenomenal, brilliant directing, and wonderful story."
        result = predict_sentiment(review, model_name=model_name)

        assert isinstance(result, dict)
        assert result["sentiment"] == "positive"
        assert result["model"] == model_name
        assert isinstance(result["confidence"], float)
        assert 0.5 <= result["confidence"] <= 1.0

    @pytest.mark.parametrize("model_name", ["baseline_nb", "baseline_logistic"])
    def test_negative_review_prediction(self, model_name: str):
        review = "A complete disaster of a film. Awful, dull, predictable, and utterly terrible acting."
        result = predict_sentiment(review, model_name=model_name)

        assert isinstance(result, dict)
        assert result["sentiment"] == "negative"
        assert result["model"] == model_name
        assert isinstance(result["confidence"], float)
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
