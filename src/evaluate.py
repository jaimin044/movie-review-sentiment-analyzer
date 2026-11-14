"""Evaluation module for Context-Aware Movie Review Sentiment Analyzer.

Loads saved model pipelines, generates predictions on the test dataset,
computes evaluation metrics (accuracy, precision, recall, F1, confusion matrices),
generates comparison tables and visualizations, and persists reports.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import REQUIRED_COLUMNS, load_dataset

# Explicit label ordering: negative at index 0, positive at index 1
LABEL_ORDER = ["negative", "positive"]

# Registry of models supported for evaluation
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "baseline_nb": {
        "name": "Baseline TF-IDF + Naive Bayes (E1)",
        "file": "baseline_nb.joblib",
    },
    "baseline_logistic": {
        "name": "Baseline TF-IDF + Logistic Regression (E2)",
        "file": "baseline_logistic.joblib",
    },
    "context_nb": {
        "name": "Context-Aware TF-IDF + Naive Bayes (E3)",
        "file": "context_nb.joblib",
    },
    "context_logistic": {
        "name": "Context-Aware TF-IDF + Logistic Regression (E4)",
        "file": "context_logistic.joblib",
    },
}


def load_test_data(test_path: str = "data/processed/test.csv") -> pd.DataFrame:
    """Loads and validates the test dataset.

    Args:
        test_path: Path to test CSV file.

    Returns:
        pd.DataFrame: Validated test DataFrame.

    Raises:
        ValueError: If required columns are missing or null values are present.
    """
    df = load_dataset(test_path)
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Test dataset is missing required columns: {sorted(list(missing_cols))}")

    if df["review"].isna().sum() > 0 or df["sentiment"].isna().sum() > 0:
        raise ValueError("Test dataset contains missing/null values in required columns.")

    valid_labels = set(LABEL_ORDER)
    found_labels = set(df["sentiment"].unique())
    if not found_labels.issubset(valid_labels):
        raise ValueError(f"Invalid labels in test dataset: {found_labels - valid_labels}")

    return df


def load_saved_pipelines(
    models_dir: str = "models",
    target_models: Optional[List[str]] = None,
) -> dict[str, dict[str, Any]]:
    """Discovers and loads available serialized model pipelines from disk.

    Args:
        models_dir: Directory containing .joblib model files.
        target_models: Optional list of specific model keys to load.

    Returns:
        dict[str, dict[str, Any]]: Mapping of model_id to model metadata and pipeline.

    Raises:
        FileNotFoundError: If no requested or registered models are found on disk.
    """
    candidates = target_models or list(MODEL_REGISTRY.keys())
    loaded_models: dict[str, dict[str, Any]] = {}

    for model_id in candidates:
        if model_id not in MODEL_REGISTRY:
            print(f"Warning: Model '{model_id}' is not in the recognized MODEL_REGISTRY; skipping.")
            continue

        meta = MODEL_REGISTRY[model_id]
        model_file = os.path.join(models_dir, meta["file"])

        if os.path.exists(model_file):
            print(f"Loading pipeline for {meta['name']} from: {model_file}")
            pipeline: Pipeline = joblib.load(model_file)
            # Ensure multi-core execution for batch evaluation if preprocessor is present
            if "preprocessor" in pipeline.named_steps:
                pipeline.named_steps["preprocessor"].n_jobs = -1

            loaded_models[model_id] = {
                "name": meta["name"],
                "file": model_file,
                "pipeline": pipeline,
            }
        else:
            print(f"Model artifact for '{model_id}' not found at {model_file} (skipping).")

    if not loaded_models:
        raise FileNotFoundError(
            f"No trained model artifacts were found in directory: '{models_dir}'. "
            "Please run 'python -m src.train' first."
        )

    return loaded_models


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_id: str,
    model_name: str,
) -> dict[str, Any]:
    """Computes standard classification evaluation metrics.

    Args:
        y_true: True ground truth labels.
        y_pred: Model predicted labels.
        model_id: Short identifier (e.g. 'baseline_nb').
        model_name: Descriptive title of the model.

    Returns:
        dict[str, Any]: Metric dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="positive", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="positive", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="positive", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "support_negative": int(report_dict["negative"]["support"]),
        "support_positive": int(report_dict["positive"]["support"]),
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str,
) -> None:
    """Plots and saves a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        model_name: Model title for figure.
        output_path: File path to save figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar=False,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Sentiment", fontsize=11, labelpad=8)
    ax.set_ylabel("True Sentiment", fontsize=11, labelpad=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {output_path}")


def plot_model_comparison(
    metrics_list: List[dict[str, Any]],
    output_path: str,
) -> None:
    """Plots and saves a grouped bar chart comparing models across key metrics.

    Args:
        metrics_list: List of metric dictionaries.
        output_path: File path to save comparison figure.
    """
    metric_keys = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data_rows = []
    for m in metrics_list:
        for key, label in zip(metric_keys, metric_labels):
            data_rows.append({
                "Model": m["model_name"],
                "Metric": label,
                "Score": m[key],
            })

    df_plot = pd.DataFrame(data_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("muted", len(metric_labels))
    sns.barplot(
        data=df_plot,
        x="Model",
        y="Score",
        hue="Metric",
        palette=palette,
        ax=ax,
    )

    ax.set_title("Model Performance Comparison (Test Dataset)", fontsize=14, pad=14, weight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylim(0.70, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(title="Metric", loc="lower right", frameon=True)

    # Annotate numeric values above bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved model comparison plot to: {output_path}")


def evaluate_all(
    test_path: str = "data/processed/test.csv",
    models_dir: str = "models",
    reports_dir: str = "reports",
    target_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Orchestrates end-to-end evaluation for all available serialized pipelines.

    Workflow:
    1. Loads test dataset without altering rows or labels.
    2. Loads serialized pipelines from disk.
    3. Runs inference on test reviews using complete saved pipelines.
    4. Computes accuracy, precision, recall, F1, and confusion matrices.
    5. Saves detailed classification reports to reports/classification_reports.txt.
    6. Saves tabular results to reports/evaluation_results.csv.
    7. Generates confusion matrix heatmaps and comparative performance plots.

    Args:
        test_path: Path to test.csv.
        models_dir: Directory containing saved models.
        reports_dir: Directory to save generated reports and plots.
        target_models: Optional specific models to evaluate.

    Returns:
        pd.DataFrame: Comparison DataFrame of evaluation metrics.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load test data
    test_df = load_test_data(test_path)
    X_test = test_df["review"].tolist()
    y_test = test_df["sentiment"].tolist()
    print(f"\nLoaded {len(test_df):,} test reviews for evaluation.")
    print(f"Test class distribution: {test_df['sentiment'].value_counts().to_dict()}")

    # 2. Load pipelines
    pipelines = load_saved_pipelines(models_dir=models_dir, target_models=target_models)

    # 3. Evaluate each model
    all_metrics: List[dict[str, Any]] = []
    reports_text: List[str] = [
        "=" * 75,
        "Context-Aware Movie Review Sentiment Analyzer - Model Evaluation Report",
        f"Test Dataset Size: {len(test_df):,} samples",
        "=" * 75,
        "",
    ]

    for model_id, model_info in pipelines.items():
        print("\n" + "-" * 70)
        print(f"Evaluating: {model_info['name']} ({model_id})")
        print("-" * 70)

        pipeline: Pipeline = model_info["pipeline"]
        y_pred = pipeline.predict(X_test)

        metrics = compute_metrics(
            y_true=y_test,
            y_pred=y_pred,
            model_id=model_id,
            model_name=model_info["name"],
        )
        all_metrics.append(metrics)

        print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(metrics["classification_report_str"])

        # Append to report document
        reports_text.append("-" * 75)
        reports_text.append(f"Model: {model_info['name']} (ID: {model_id})")
        reports_text.append(f"Artifact: {model_info['file']}")
        reports_text.append("-" * 75)
        reports_text.append(f"Accuracy:  {metrics['accuracy']:.4f}")
        reports_text.append(f"Precision: {metrics['precision']:.4f} (positive class)")
        reports_text.append(f"Recall:    {metrics['recall']:.4f} (positive class)")
        reports_text.append(f"F1-Score:  {metrics['f1_score']:.4f} (positive class)")
        reports_text.append(f"Macro F1:  {metrics['macro_f1']:.4f}")
        reports_text.append(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        reports_text.append("\nClassification Report:")
        reports_text.append(metrics["classification_report_str"])
        reports_text.append("\nConfusion Matrix [negative, positive]:")
        reports_text.append(str(metrics["confusion_matrix"]))
        reports_text.append("\n\n")

        # Save individual confusion matrix plot
        cm_filename = f"confusion_matrix_{model_id}.png"
        cm_path = os.path.join(reports_dir, cm_filename)
        plot_confusion_matrix(
            cm=metrics["confusion_matrix"],
            model_name=model_info["name"],
            output_path=cm_path,
        )

    # 4. Save textual reports
    reports_file_path = os.path.join(reports_dir, "classification_reports.txt")
    with open(reports_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(reports_text))
    print(f"\nSaved combined classification reports to: {reports_file_path}")

    # 5. Build and save comparison CSV
    summary_rows = [
        {
            "model_id": m["model_id"],
            "model_name": m["model_name"],
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1_score"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "weighted_f1": round(m["weighted_f1"], 4),
            "support_negative": m["support_negative"],
            "support_positive": m["support_positive"],
        }
        for m in all_metrics
    ]
    comparison_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(reports_dir, "evaluation_results.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results table to: {csv_path}")

    # 6. Save comparison plot
    comparison_plot_path = os.path.join(reports_dir, "model_comparison.png")
    plot_model_comparison(all_metrics, comparison_plot_path)

    print("\n" + "=" * 70)
    print("Evaluation Summary Table:")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    return comparison_df


def main() -> None:
    """CLI execution entrypoint."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained sentiment analysis models on test dataset."
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/test.csv",
        help="Path to test data CSV (default: data/processed/test.csv)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing saved model pipelines (default: models)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory to save evaluation reports and plots (default: reports)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific model IDs to evaluate (default: all available)",
    )
    args = parser.parse_args()

    evaluate_all(
        test_path=args.test_path,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        target_models=args.models,
    )


if __name__ == "__main__":
    main()
