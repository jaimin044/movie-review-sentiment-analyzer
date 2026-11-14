"""Model training module for Context-Aware Movie Review Sentiment Analyzer.

Trains, evaluates, and serializes baseline machine learning models:
- Experiment 1 (E1): Baseline TF-IDF + Multinomial Naive Bayes
- Experiment 2 (E2): Baseline TF-IDF + Logistic Regression

Saves complete, self-contained end-to-end inference pipelines:
    Raw review -> TextPreprocessor -> TfidfVectorizer -> Classifier -> Sentiment
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import REQUIRED_COLUMNS, load_dataset
from src.vectorizer import (
    TextPreprocessor,
    build_baseline_pipeline,
    create_baseline_vectorizer,
    create_preprocessor,
)

# Canonical label ordering
LABEL_ORDER = ["negative", "positive"]


def validate_dataset_schema(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> None:
    """Validates schema, null values, and sentiment labels of a dataset.

    Args:
        df: Input DataFrame to validate.
        dataset_name: Label used in error/log messages.

    Raises:
        ValueError: If required columns are missing, nulls exist, or labels are invalid.
    """
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(list(missing_cols))}. "
            f"Available: {df.columns.tolist()}"
        )

    null_review_count = df["review"].isna().sum()
    null_sentiment_count = df["sentiment"].isna().sum()
    if null_review_count > 0 or null_sentiment_count > 0:
        raise ValueError(
            f"{dataset_name} contains null values: "
            f"{null_review_count} in 'review', {null_sentiment_count} in 'sentiment'."
        )

    unique_labels = set(df["sentiment"].unique())
    expected_labels = set(LABEL_ORDER)
    if not unique_labels.issubset(expected_labels):
        invalid = unique_labels - expected_labels
        raise ValueError(
            f"{dataset_name} contains invalid sentiment labels: {invalid}. "
            f"Expected only: {expected_labels}"
        )


def load_and_validate_data(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and validates both training and testing datasets.

    Args:
        train_path: Path to training CSV file.
        test_path: Path to testing CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    print(f"Loading training data from: {train_path}")
    train_df = load_dataset(train_path)
    validate_dataset_schema(train_df, "Training dataset")

    print(f"Loading testing data from: {test_path}")
    test_df = load_dataset(test_path)
    validate_dataset_schema(test_df, "Testing dataset")

    train_dist = train_df["sentiment"].value_counts().to_dict()
    test_dist = test_df["sentiment"].value_counts().to_dict()

    print(f"Loaded {len(train_df):,} training samples: {train_dist}")
    print(f"Loaded {len(test_df):,} testing samples: {test_dist}")

    return train_df, test_df


def get_preprocessed_texts(
    texts: Sequence[str],
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    n_jobs: int = -1,
    description: str = "dataset",
) -> list[str]:
    """Preprocesses a sequence of texts, using cache if available.

    Args:
        texts: Raw text documents.
        cache_path: Optional CSV file path to load/store cached preprocessed texts.
        force_recompute: If True, ignores existing cache and recomputes.
        n_jobs: Number of parallel workers for preprocessing. Defaults to -1 (all cores).
        description: Description of data for log output.

    Returns:
        list[str]: Preprocessed text strings.
    """
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print(f"Found preprocessed cache for {description} at: {cache_path}")
        cache_df = pd.read_csv(cache_path)
        if "clean_review" in cache_df.columns and len(cache_df) == len(texts):
            print(f"Loaded {len(cache_df):,} preprocessed texts from cache.")
            return cache_df["clean_review"].fillna("").astype(str).tolist()
        else:
            print("Cache is invalid or length mismatch. Recomputing...")

    start_time = time.time()
    print(f"Preprocessing {description} ({len(texts):,} reviews) using {n_jobs} workers...")
    preprocessor = TextPreprocessor(n_jobs=n_jobs)
    clean_texts = preprocessor.transform(texts)
    elapsed = time.time() - start_time
    print(f"Finished preprocessing {description} in {elapsed:.2f}s ({elapsed / len(texts):.4f}s/doc).")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        pd.DataFrame({"clean_review": clean_texts}).to_csv(cache_path, index=False)
        print(f"Cached preprocessed texts to: {cache_path}")

    return clean_texts


def train_baseline_models(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    models_dir: str = "models",
    train_cache_path: str = "data/processed/train_preprocessed.csv",
    test_cache_path: str = "data/processed/test_preprocessed.csv",
    force_recompute: bool = False,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Trains, evaluates, and serializes baseline models E1 and E2.

    Workflow:
    1. Load and validate train.csv and test.csv.
    2. Preprocess text data (cached for efficiency).
    3. Fit baseline TF-IDF vectorizer only on training data.
    4. Train E1: TF-IDF + Multinomial Naive Bayes (alpha=1.0).
    5. Train E2: TF-IDF + Logistic Regression (C=1.0, max_iter=1000, random_state=42).
    6. Package complete scikit-learn Pipelines:
       [preprocessor -> tfidf -> classifier].
    7. Serialize to models/baseline_nb.joblib and models/baseline_logistic.joblib.
    8. Print performance metrics and sample predictions.

    Args:
        train_path: Path to train.csv.
        test_path: Path to test.csv.
        models_dir: Directory where serialized models will be saved.
        train_cache_path: Path to cached preprocessed training data.
        test_cache_path: Path to cached preprocessed testing data.
        force_recompute: Whether to ignore preprocessed cache.
        n_jobs: Number of parallel jobs for preprocessing.

    Returns:
        dict[str, Any]: Dictionary containing trained pipelines, vectorizer, and evaluation metrics.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load and validate data
    train_df, test_df = load_and_validate_data(train_path, test_path)
    X_train_raw = train_df["review"].tolist()
    y_train = train_df["sentiment"].tolist()
    X_test_raw = test_df["review"].tolist()
    y_test = test_df["sentiment"].tolist()

    # 2. Preprocess texts
    X_train_clean = get_preprocessed_texts(
        X_train_raw,
        cache_path=train_cache_path,
        force_recompute=force_recompute,
        n_jobs=n_jobs,
        description="training data",
    )
    X_test_clean = get_preprocessed_texts(
        X_test_raw,
        cache_path=test_cache_path,
        force_recompute=force_recompute,
        n_jobs=n_jobs,
        description="testing data",
    )

    # 3. Fit baseline TF-IDF vectorizer ONLY on training data
    print("\nFitting baseline TF-IDF vectorizer on training data...")
    tfidf_start = time.time()
    vectorizer = create_baseline_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train_clean)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    print(f"TF-IDF fitting complete in {time.time() - tfidf_start:.2f}s.")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features.")
    print(f"Training TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Testing TF-IDF matrix shape:  {X_test_tfidf.shape}")

    results: dict[str, Any] = {}

    # Inference preprocessor step for saved pipelines (n_jobs=1 for fast inference)
    inference_preprocessor = create_preprocessor(n_jobs=1)

    # -------------------------------------------------------------------------
    # Experiment 1 (E1): Baseline TF-IDF + Multinomial Naive Bayes
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training Experiment 1 (E1): Baseline TF-IDF + Multinomial Naive Bayes")
    print("=" * 70)
    nb_start = time.time()
    nb_clf = MultinomialNB(alpha=1.0)
    nb_clf.fit(X_train_tfidf, y_train)
    nb_train_time = time.time() - nb_start

    # Assemble complete end-to-end pipeline
    pipeline_nb = Pipeline(
        steps=[
            ("preprocessor", inference_preprocessor),
            ("tfidf", vectorizer),
            ("classifier", nb_clf),
        ]
    )

    # Evaluate E1
    y_pred_nb = nb_clf.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    prec_nb = precision_score(y_test, y_pred_nb, pos_label="positive", zero_division=0)
    rec_nb = recall_score(y_test, y_pred_nb, pos_label="positive", zero_division=0)
    f1_nb = f1_score(y_test, y_pred_nb, pos_label="positive", zero_division=0)

    print(f"E1 Training time: {nb_train_time:.2f}s")
    print(f"E1 Test Accuracy:  {acc_nb:.4f} ({acc_nb * 100:.2f}%)")
    print(f"E1 Test Precision: {prec_nb:.4f}")
    print(f"E1 Test Recall:    {rec_nb:.4f}")
    print(f"E1 Test F1-Score:  {f1_nb:.4f}")
    print("\nClassification Report (E1 - Multinomial NB):")
    print(classification_report(y_test, y_pred_nb, labels=LABEL_ORDER, digits=4, zero_division=0))

    # Serialize E1 pipeline
    nb_model_path = os.path.join(models_dir, "baseline_nb.joblib")
    joblib.dump(pipeline_nb, nb_model_path)
    print(f"Saved E1 pipeline to: {nb_model_path}")

    results["baseline_nb"] = {
        "pipeline": pipeline_nb,
        "model_path": nb_model_path,
        "accuracy": acc_nb,
        "precision": prec_nb,
        "recall": rec_nb,
        "f1": f1_nb,
    }

    # -------------------------------------------------------------------------
    # Experiment 2 (E2): Baseline TF-IDF + Logistic Regression
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Training Experiment 2 (E2): Baseline TF-IDF + Logistic Regression")
    print("=" * 70)
    lr_start = time.time()
    lr_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr_clf.fit(X_train_tfidf, y_train)
    lr_train_time = time.time() - lr_start

    # Assemble complete end-to-end pipeline
    pipeline_lr = Pipeline(
        steps=[
            ("preprocessor", inference_preprocessor),
            ("tfidf", vectorizer),
            ("classifier", lr_clf),
        ]
    )

    # Evaluate E2
    y_pred_lr = lr_clf.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, pos_label="positive", zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, pos_label="positive", zero_division=0)
    f1_lr = f1_score(y_test, y_pred_lr, pos_label="positive", zero_division=0)

    print(f"E2 Training time: {lr_train_time:.2f}s")
    print(f"E2 Test Accuracy:  {acc_lr:.4f} ({acc_lr * 100:.2f}%)")
    print(f"E2 Test Precision: {prec_lr:.4f}")
    print(f"E2 Test Recall:    {rec_lr:.4f}")
    print(f"E2 Test F1-Score:  {f1_lr:.4f}")
    print("\nClassification Report (E2 - Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, labels=LABEL_ORDER, digits=4, zero_division=0))

    # Serialize E2 pipeline
    lr_model_path = os.path.join(models_dir, "baseline_logistic.joblib")
    joblib.dump(pipeline_lr, lr_model_path)
    print(f"Saved E2 pipeline to: {lr_model_path}")

    results["baseline_logistic"] = {
        "pipeline": pipeline_lr,
        "model_path": lr_model_path,
        "accuracy": acc_lr,
        "precision": prec_lr,
        "recall": rec_lr,
        "f1": f1_lr,
    }

    return results


def verify_serialized_pipelines(models_dir: str = "models") -> None:
    """Loads serialized pipelines and tests inference on sample raw reviews.

    Args:
        models_dir: Directory containing saved .joblib files.
    """
    print("\n" + "=" * 70)
    print("Verifying Serialized Model Pipelines on Raw Review Inputs")
    print("=" * 70)

    sample_reviews = [
        "An absolute masterpiece! The acting and directing were phenomenal and deeply moving.",
        "A complete disaster. Dull, predictable, and awful acting throughout the film.",
        "The movie was not good at all, completely lacked any emotional depth or spark.",
        "The acting was surprisingly good, but the screenplay was a terrible mess.",
    ]

    for model_name, filename in [
        ("Baseline Naive Bayes (E1)", "baseline_nb.joblib"),
        ("Baseline Logistic Regression (E2)", "baseline_logistic.joblib"),
    ]:
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            print(f"[ERROR] Model artifact missing at: {model_path}")
            continue

        print(f"\nTesting loaded pipeline from: {model_path} ({model_name})")
        loaded_pipe = joblib.load(model_path)
        predictions = loaded_pipe.predict(sample_reviews)
        probabilities = loaded_pipe.predict_proba(sample_reviews)
        classes = list(loaded_pipe.classes_)

        for rev, pred, proba in zip(sample_reviews, predictions, probabilities):
            pos_idx = classes.index("positive")
            confidence = proba[pos_idx] if pred == "positive" else proba[classes.index("negative")]
            print(f"  Input: \"{rev}\"")
            print(f"    -> Prediction: {pred.upper()} (Confidence: {confidence:.2%})")
            print(f"    -> Probabilities: {dict(zip(classes, [round(p, 4) for p in proba]))}")


def main() -> None:
    """Main execution function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and serialize baseline sentiment analysis models (E1 & E2)."
    )
    parser.add_argument(
        "--train-path",
        default="data/processed/train.csv",
        help="Path to training data CSV (default: data/processed/train.csv)",
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/test.csv",
        help="Path to testing data CSV (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save serialized models (default: models)",
    )
    parser.add_argument(
        "--train-cache",
        default="data/processed/train_preprocessed.csv",
        help="Cache path for preprocessed training reviews",
    )
    parser.add_argument(
        "--test-cache",
        default="data/processed/test_preprocessed.csv",
        help="Cache path for preprocessed testing reviews",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached preprocessed texts and recompute",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel worker jobs for preprocessing (default: -1)",
    )
    args = parser.parse_args()

    train_baseline_models(
        train_path=args.train_path,
        test_path=args.test_path,
        models_dir=args.models_dir,
        train_cache_path=args.train_cache,
        test_cache_path=args.test_cache,
        force_recompute=args.force_recompute,
        n_jobs=args.n_jobs,
    )

    verify_serialized_pipelines(models_dir=args.models_dir)


if __name__ == "__main__":
    main()
