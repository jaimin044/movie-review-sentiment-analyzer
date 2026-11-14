"""Unit tests for the evaluation module and metrics generation."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    LABEL_ORDER,
    compute_metrics,
    load_saved_pipelines,
    load_test_data,
    plot_confusion_matrix,
    plot_model_comparison,
)


class TestEvaluationMetrics:
    """Tests for metric calculation and output formatting."""

    def test_compute_metrics_perfect_predictions(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "positive", "negative", "negative"]

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            model_id="test_model",
            model_name="Test Model",
        )

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0
        assert metrics["support_positive"] == 2
        assert metrics["support_negative"] == 2
        assert isinstance(metrics["confusion_matrix"], np.ndarray)
        assert np.array_equal(metrics["confusion_matrix"], np.array([[2, 0], [0, 2]]))

    def test_compute_metrics_mixed_predictions(self):
        y_true = ["positive", "positive", "negative", "negative"]
        y_pred = ["positive", "negative", "negative", "positive"]

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            model_id="test_model",
            model_name="Test Model",
        )

        assert metrics["accuracy"] == 0.5
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0


class TestEvaluationPlotting:
    """Tests for generating confusion matrix and comparison charts."""

    def test_plot_confusion_matrix(self):
        cm = np.array([[10, 2], [1, 15]])
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            plot_confusion_matrix(cm, "Test Model", tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0

    def test_plot_model_comparison(self):
        metrics_list = [
            {
                "model_name": "Model A",
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.86,
                "f1_score": 0.85,
            },
            {
                "model_name": "Model B",
                "accuracy": 0.89,
                "precision": 0.88,
                "recall": 0.90,
                "f1_score": 0.89,
            },
        ]
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            plot_model_comparison(metrics_list, tmp.name)
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0


class TestPipelineDiscovery:
    """Tests for discovering and loading saved pipelines."""

    def test_load_saved_pipelines_existing_models(self):
        pipelines = load_saved_pipelines(models_dir="models")
        assert "baseline_nb" in pipelines
        assert "baseline_logistic" in pipelines
        assert "pipeline" in pipelines["baseline_nb"]
        assert "pipeline" in pipelines["baseline_logistic"]

    def test_load_saved_pipelines_nonexistent_dir_raises(self):
        with tempfile.TemporaryDirectory() as empty_dir:
            with pytest.raises(FileNotFoundError, match="No trained model artifacts"):
                load_saved_pipelines(models_dir=empty_dir)
