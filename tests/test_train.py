"""Unit tests for training module and data validation."""

import os
import tempfile
import pandas as pd
import pytest

from src.train import (
    LABEL_ORDER,
    load_and_validate_data,
    train_baseline_models,
    train_context_models,
    validate_dataset_schema,
)


class TestDataValidation:
    """Tests for dataset schema, nulls, and label validation."""

    def test_valid_dataset_schema(self):
        df = pd.DataFrame({
            "review": ["Great movie!", "Terrible film."],
            "sentiment": ["positive", "negative"],
        })
        # Should not raise
        validate_dataset_schema(df, "Test set")

    def test_missing_column(self):
        df = pd.DataFrame({"review": ["Great movie!"]})
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataset_schema(df, "Test set")

    def test_null_values(self):
        df = pd.DataFrame({
            "review": ["Great movie!", None],
            "sentiment": ["positive", "negative"],
        })
        with pytest.raises(ValueError, match="contains null values"):
            validate_dataset_schema(df, "Test set")

    def test_invalid_sentiment_labels(self):
        df = pd.DataFrame({
            "review": ["Great movie!", "Terrible film."],
            "sentiment": ["positive", "neutral"],
        })
        with pytest.raises(ValueError, match="invalid sentiment labels"):
            validate_dataset_schema(df, "Test set")


class TestBaselineTrainingMini:
    """Tests training flow on a mini dataset."""

    def test_mini_training_and_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")
            models_dir = os.path.join(tmpdir, "models")

            train_df = pd.DataFrame({
                "review": [
                    "This is an amazing and fantastic film! Truly fantastic and wonderful.",
                    "Loved the amazing film, wonderful acting, absolutely fantastic.",
                    "Truly a wonderful experience, brilliantly directed and fantastic.",
                    "Terrible, awful, horrible worst film I have ever seen.",
                    "Boring, terrible, awful, horrible waste of time.",
                    "Hated every single second of this awful terrible horrible film.",
                ],
                "sentiment": ["positive", "positive", "positive", "negative", "negative", "negative"],
            })
            test_df = pd.DataFrame({
                "review": [
                    "Fantastic and wonderful film, truly amazing!",
                    "Terrible, awful, and horrible movie.",
                ],
                "sentiment": ["positive", "negative"],
            })

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            results = train_baseline_models(
                train_path=train_path,
                test_path=test_path,
                models_dir=models_dir,
                train_cache_path=os.path.join(tmpdir, "train_clean.csv"),
                test_cache_path=os.path.join(tmpdir, "test_clean.csv"),
                force_recompute=True,
                n_jobs=1,
            )

            assert "baseline_nb" in results
            assert "baseline_logistic" in results

            nb_path = os.path.join(models_dir, "baseline_nb.joblib")
            lr_path = os.path.join(models_dir, "baseline_logistic.joblib")

            assert os.path.exists(nb_path)
            assert os.path.exists(lr_path)

            nb_pipe = results["baseline_nb"]["pipeline"]
            lr_pipe = results["baseline_logistic"]["pipeline"]

            # Test inference on raw review
            assert nb_pipe.predict(["A brilliant and wonderful film!"])[0] == "positive"
            assert lr_pipe.predict(["A terrible, dull, and awful movie."])[0] == "negative"

    def test_mini_context_training_and_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            test_path = os.path.join(tmpdir, "test.csv")
            models_dir = os.path.join(tmpdir, "models")

            train_df = pd.DataFrame({
                "review": [
                    "This is an amazing and fantastic film! Truly fantastic and wonderful.",
                    "Loved the amazing film, wonderful acting, absolutely fantastic.",
                    "Truly a wonderful experience, brilliantly directed and fantastic.",
                    "Terrible, awful, horrible worst film I have ever seen.",
                    "Boring, terrible, awful, horrible waste of time.",
                    "Hated every single second of this awful terrible horrible film.",
                ],
                "sentiment": ["positive", "positive", "positive", "negative", "negative", "negative"],
            })
            test_df = pd.DataFrame({
                "review": [
                    "Fantastic and wonderful film, truly amazing!",
                    "Terrible, awful, and horrible movie.",
                ],
                "sentiment": ["positive", "negative"],
            })

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            results = train_context_models(
                train_path=train_path,
                test_path=test_path,
                models_dir=models_dir,
                train_cache_path=os.path.join(tmpdir, "train_clean.csv"),
                test_cache_path=os.path.join(tmpdir, "test_clean.csv"),
                force_recompute=True,
                n_jobs=1,
            )

            assert "context_nb" in results
            assert "context_logistic" in results

            nb_path = os.path.join(models_dir, "context_nb.joblib")
            lr_path = os.path.join(models_dir, "context_logistic.joblib")

            assert os.path.exists(nb_path)
            assert os.path.exists(lr_path)

            nb_pipe = results["context_nb"]["pipeline"]
            lr_pipe = results["context_logistic"]["pipeline"]

            assert nb_pipe.predict(["A brilliant and wonderful film!"])[0] == "positive"
            assert lr_pipe.predict(["A terrible, dull, and awful movie."])[0] == "negative"
