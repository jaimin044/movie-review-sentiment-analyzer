"""Unit tests for the Streamlit application logic and helper functions.

Covers:
- Input validation (empty, whitespace, None, short review warning)
- Prediction history formatting, appending, and rerun duplicate prevention
- Session history clearing and CSV export generation
- Dashboard metrics data preparation, percentage scaling, and best model identification
- Missing evaluation CSV, empty reports, and missing column handling
- Missing confusion matrix detection
- Model artifact availability discovery
"""

from pathlib import Path
import pandas as pd
import pytest

from app import (
    add_prediction_to_history,
    check_confusion_matrix_status,
    export_history_to_csv,
    format_prediction_record,
    prepare_dashboard_data,
    validate_review_input,
    CONFUSION_FILES,
    MODEL_INFO,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.predict import list_available_models


class TestReviewInputValidation:
    """Tests for review input quality checks and warning generation."""

    def test_valid_standard_review(self):
        text = "This is a great movie with wonderful acting and music."
        is_valid, status, msg = validate_review_input(text)
        assert is_valid is True
        assert status == "valid"
        assert msg is None

    def test_empty_string_rejected(self):
        is_valid, status, msg = validate_review_input("")
        assert is_valid is False
        assert status == "empty"
        assert "non-empty" in msg

    def test_whitespace_only_rejected(self):
        is_valid, status, msg = validate_review_input("   \n\t   ")
        assert is_valid is False
        assert status == "empty"

    def test_none_input_rejected(self):
        is_valid, status, msg = validate_review_input(None)
        assert is_valid is False
        assert status == "empty"

    def test_short_review_warning_generated(self):
        # Under 5 words should be valid but trigger a short review informational warning
        text = "Not bad."
        is_valid, status, msg = validate_review_input(text)
        assert is_valid is True
        assert status == "short"
        assert "Short Review Notice" in msg
        assert "informational" in msg

    def test_boundary_five_words_no_warning(self):
        text = "One two three four five."
        is_valid, status, msg = validate_review_input(text)
        assert is_valid is True
        assert status == "valid"
        assert msg is None


class TestPredictionHistoryManagement:
    """Tests for prediction history formatting, deduplication, and export."""

    def test_format_prediction_record(self):
        rec = format_prediction_record(
            review="A fantastic masterpiece of cinema!",
            model_id="baseline_logistic",
            model_name="E2 - Baseline TF-IDF + Logistic Regression",
            sentiment="positive",
            confidence=0.8945,
            timestamp="2026-11-15 10:00:00",
        )
        assert rec["timestamp"] == "2026-11-15 10:00:00"
        assert rec["model_id"] == "baseline_logistic"
        assert rec["sentiment"] == "Positive"
        assert rec["confidence"] == 0.8945
        assert rec["confidence_pct"] == "89.45%"
        assert rec["review_preview"] == "A fantastic masterpiece of cinema!"

    def test_format_prediction_record_preview_truncation(self):
        long_review = "Word " * 50
        rec = format_prediction_record(
            review=long_review,
            model_id="baseline_nb",
            model_name="E1",
            sentiment="negative",
            confidence=0.75,
        )
        assert len(rec["review_preview"]) <= 90
        assert rec["review_preview"].endswith("...")

    def test_history_insertion_and_duplicate_prevention(self):
        history = []
        rec1 = format_prediction_record(
            review="Great film!",
            model_id="baseline_logistic",
            model_name="E2",
            sentiment="positive",
            confidence=0.88,
            timestamp="2026-11-15 10:05:00",
        )
        history = add_prediction_to_history(history, rec1)
        assert len(history) == 1

        # Attempt to add exact duplicate (same review, model_id, timestamp)
        history = add_prediction_to_history(history, rec1)
        assert len(history) == 1, "Duplicate record was erroneously appended on rerun!"

        # Add a different review
        rec2 = format_prediction_record(
            review="Awful movie!",
            model_id="baseline_logistic",
            model_name="E2",
            sentiment="negative",
            confidence=0.92,
            timestamp="2026-11-15 10:06:00",
        )
        history = add_prediction_to_history(history, rec2)
        assert len(history) == 2

    def test_history_clearing(self):
        history = [{"dummy": "record"}]
        history.clear()
        assert len(history) == 0

    def test_export_history_to_csv(self):
        history = [
            format_prediction_record(
                review="Terrific visual storytelling.",
                model_id="baseline_logistic",
                model_name="E2 - Baseline Logistic",
                sentiment="positive",
                confidence=0.9123,
                timestamp="2026-11-15 10:10:00",
            )
        ]
        csv_str = export_history_to_csv(history)
        assert "timestamp,model_id,model_name,sentiment,confidence,confidence_pct,review" in csv_str
        assert "Terrific visual storytelling." in csv_str
        assert "baseline_logistic" in csv_str
        assert "91.23%" in csv_str

    def test_export_empty_history_returns_empty_string(self):
        assert export_history_to_csv([]) == ""


class TestDashboardDataPreparation:
    """Tests for dashboard metrics parsing, percentage scaling, and error handling."""

    def test_valid_evaluation_data_preparation(self):
        raw_df = pd.DataFrame([
            {"model_id": "baseline_nb", "accuracy": 0.8613, "precision": 0.8726, "recall": 0.8475, "f1_score": 0.8599},
            {"model_id": "baseline_logistic", "accuracy": 0.8929, "precision": 0.8848, "recall": 0.9044, "f1_score": 0.8945},
            {"model_id": "context_nb", "accuracy": 0.8580, "precision": 0.8705, "recall": 0.8425, "f1_score": 0.8562},
            {"model_id": "context_logistic", "accuracy": 0.8899, "precision": 0.8814, "recall": 0.9019, "f1_score": 0.8916},
        ])
        result = prepare_dashboard_data(raw_df)

        assert result["is_valid"] is True
        assert result["error"] is None

        # Verify best model correctly identified
        assert result["best_model"]["model_id"] == "baseline_logistic"
        assert pytest.approx(result["best_model"]["accuracy"], 0.01) == 89.29

        # Verify display dataframe has percentage scale
        display_df = result["display_df"]
        assert "Accuracy (%)" in display_df.columns
        assert pytest.approx(display_df.loc[display_df["Model ID"] == "baseline_logistic", "Accuracy (%)"].values[0], 0.01) == 89.29

        # Verify melted dataframe for grouped bar chart
        melted = result["melted_df"]
        assert set(melted["Metric"].unique()) == {"Accuracy", "Precision", "Recall", "F1-Score"}
        assert len(melted) == 16  # 4 models * 4 metrics

    def test_missing_evaluation_csv_handled(self):
        result = prepare_dashboard_data(None)
        assert result["is_valid"] is False
        assert "not found" in result["error"]

    def test_empty_evaluation_csv_handled(self):
        result = prepare_dashboard_data(pd.DataFrame())
        assert result["is_valid"] is False
        assert "empty" in result["error"]

    def test_missing_columns_in_evaluation_csv_handled(self):
        bad_df = pd.DataFrame([{"model_id": "m1", "accuracy": 0.85}])
        result = prepare_dashboard_data(bad_df)
        assert result["is_valid"] is False
        assert "missing required columns" in result["error"]


class TestModelArtifactsAndReportFiles:
    """Tests for model availability and confusion matrix artifact discovery."""

    def test_all_four_models_present(self):
        available = list_available_models(models_dir=str(MODELS_DIR))
        assert len(available) == 4
        assert set(available) == {"baseline_nb", "baseline_logistic", "context_nb", "context_logistic"}

    def test_confusion_matrix_images_present(self):
        status = check_confusion_matrix_status(REPORTS_DIR)
        for mid, img_path in status.items():
            assert img_path is not None, f"Missing confusion matrix image for {mid}"
            assert img_path.exists(), f"Image path {img_path} does not exist"

    def test_confusion_matrix_empty_dir_handled(self, tmp_path):
        status = check_confusion_matrix_status(tmp_path)
        for mid, img_path in status.items():
            assert img_path is None

