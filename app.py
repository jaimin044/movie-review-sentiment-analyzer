"""Movie Review Sentiment Analyzer - Streamlit Application.

An interactive academic web application comparing baseline and context-aware
machine learning models for binary sentiment classification on the IMDb 50K
Movie Reviews dataset.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from src.predict import list_available_models, predict_sentiment

# --------------------------------------------------
# Configuration & Constants
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_INFO: dict[str, dict[str, str]] = {
    "baseline_nb": {
        "name": "E1 - Baseline TF-IDF + Multinomial Naive Bayes",
        "short_name": "E1: Baseline Naive Bayes",
        "description": "Standard unigram TF-IDF feature extraction with Multinomial Naive Bayes classifier.",
        "family": "Baseline",
    },
    "baseline_logistic": {
        "name": "E2 - Baseline TF-IDF + Logistic Regression",
        "short_name": "E2: Baseline Logistic Regression",
        "description": "Standard unigram TF-IDF feature extraction with L2-regularized Logistic Regression.",
        "family": "Baseline",
    },
    "context_nb": {
        "name": "E3 - Context-Aware TF-IDF + Multinomial Naive Bayes",
        "short_name": "E3: Context-Aware Naive Bayes",
        "description": "Context-aware TF-IDF (negation NOT_ and contrast BEFORE_BUT_/AFTER_BUT_ markers) with Multinomial Naive Bayes.",
        "family": "Context-Aware",
    },
    "context_logistic": {
        "name": "E4 - Context-Aware TF-IDF + Logistic Regression",
        "short_name": "E4: Context-Aware Logistic Regression",
        "description": "Context-aware TF-IDF (negation NOT_ and contrast BEFORE_BUT_/AFTER_BUT_ markers) with Logistic Regression.",
        "family": "Context-Aware",
    },
}

CONFUSION_FILES: dict[str, str] = {
    "baseline_nb": "confusion_matrix_baseline_nb.png",
    "baseline_logistic": "confusion_matrix_baseline_logistic.png",
    "context_nb": "confusion_matrix_context_nb.png",
    "context_logistic": "confusion_matrix_context_logistic.png",
}

SAMPLE_REVIEWS: dict[str, str] = {
    "Clearly Positive": (
        "An absolute masterpiece! The acting and directing were phenomenal, "
        "emotionally resonant, and visually stunning throughout."
    ),
    "Clearly Negative": (
        "A complete disaster. Dull, predictable, and awful acting throughout the film. "
        "Not worth a single minute of your time."
    ),
    "Negation Heavy": (
        "The movie was not good at all, completely lacked any spark, "
        "and I cannot recommend it to anyone looking for genuine entertainment."
    ),
    "Contrastive Shift ('but')": (
        "The cinematography was breathtaking and the soundtrack was wonderful, "
        "but the plot was painfully boring and the dialogues were terrible."
    ),
    "Very Short": (
        "Not bad."
    ),
}

# --------------------------------------------------
# Pure Helper Functions (Modular & Testable)
# --------------------------------------------------

def validate_review_input(text: Any) -> Tuple[bool, str, Optional[str]]:
    """Validates user review text input.

    Args:
        text: Raw user input text.

    Returns:
        Tuple[bool, str, Optional[str]]: (is_valid, status_code, message)
            status_code is one of: 'valid', 'empty', 'short'.
    """
    if text is None or not isinstance(text, str):
        return False, "empty", "Please enter a valid movie review string."

    trimmed = text.strip()
    if not trimmed:
        return False, "empty", "Please enter a non-empty movie review to analyze."

    word_count = len(trimmed.split())
    if word_count < 5:
        msg = (
            "ℹ️ **Short Review Notice**: Reviews under 5 words provide minimal contextual signals. "
            "Predictions on extremely brief phrases are informational and may have lower reliability."
        )
        return True, "short", msg

    return True, "valid", None


def format_prediction_record(
    review: str,
    model_id: str,
    model_name: str,
    sentiment: str,
    confidence: float,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Formats a sentiment prediction into a standardized session record.

    Args:
        review: Original user review text.
        model_id: Model identifier (e.g. 'baseline_logistic').
        model_name: Full descriptive model name.
        sentiment: Predicted sentiment ('positive' or 'negative').
        confidence: Posterior probability score between 0.0 and 1.0.
        timestamp: Optional formatted timestamp string.

    Returns:
        Dict[str, Any]: Formatted history dictionary.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_sentiment = sentiment.strip().capitalize()
    conf_float = round(float(confidence), 4)

    preview = review.strip()
    if len(preview) > 85:
        preview = preview[:85].rstrip() + "..."

    return {
        "timestamp": ts,
        "model_id": model_id,
        "model_name": model_name,
        "sentiment": clean_sentiment,
        "confidence": conf_float,
        "confidence_pct": f"{conf_float * 100:.2f}%",
        "review": review.strip(),
        "review_preview": preview,
    }


def add_prediction_to_history(
    history: List[Dict[str, Any]],
    new_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Appends a new prediction record while preventing duplicate entries from page reruns.

    Args:
        history: Current prediction history list.
        new_record: New formatted prediction record.

    Returns:
        List[Dict[str, Any]]: Updated history list.
    """
    if history:
        last = history[-1]
        # Prevent identical duplicate if review, model, and timestamp are the same
        if (
            last.get("review") == new_record.get("review")
            and last.get("model_id") == new_record.get("model_id")
            and last.get("timestamp") == new_record.get("timestamp")
        ):
            return history

    history.append(new_record)
    return history


def export_history_to_csv(history: List[Dict[str, Any]]) -> str:
    """Exports prediction history list to CSV formatted string.

    Args:
        history: List of prediction dictionaries.

    Returns:
        str: CSV string representation.
    """
    if not history:
        return ""

    export_cols = [
        "timestamp",
        "model_id",
        "model_name",
        "sentiment",
        "confidence",
        "confidence_pct",
        "review",
    ]
    df = pd.DataFrame(history)
    available_cols = [c for c in export_cols if c in df.columns]
    return df[available_cols].to_csv(index=False)


def prepare_dashboard_data(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Prepares and validates evaluation metrics dataframe for dashboard display.

    Args:
        df: Raw evaluation results dataframe.

    Returns:
        Dict[str, Any]: Formatted data dictionary containing:
            - 'is_valid': bool
            - 'error': Optional[str]
            - 'display_df': Optional[pd.DataFrame] with percentage formatting
            - 'melted_df': Optional[pd.DataFrame] for grouped bar charting
            - 'best_model': Optional[pd.Series]
    """
    if df is None:
        return {
            "is_valid": False,
            "error": "Evaluation report file not found. Run 'python -m src.evaluate' first.",
            "display_df": None,
            "melted_df": None,
            "best_model": None,
        }

    if df.empty:
        return {
            "is_valid": False,
            "error": "Evaluation report is empty.",
            "display_df": None,
            "melted_df": None,
            "best_model": None,
        }

    required_cols = {"model_id", "accuracy", "precision", "recall", "f1_score"}
    missing = required_cols - set(df.columns)
    if missing:
        return {
            "is_valid": False,
            "error": f"Evaluation report missing required columns: {sorted(list(missing))}",
            "display_df": None,
            "melted_df": None,
            "best_model": None,
        }

    cleaned = df.copy()

    # Map full descriptive names
    cleaned["model_name"] = cleaned["model_id"].map(
        lambda mid: MODEL_INFO.get(mid, {}).get("name", mid)
    )
    cleaned["short_name"] = cleaned["model_id"].map(
        lambda mid: MODEL_INFO.get(mid, {}).get("short_name", mid)
    )

    # Convert to percentage scale 0-100
    metric_cols = ["accuracy", "precision", "recall", "f1_score"]
    for col in metric_cols:
        cleaned[col] = cleaned[col].astype(float) * 100.0

    best_idx = cleaned["accuracy"].idxmax()
    best_model = cleaned.loc[best_idx]

    # Display dataframe with model ID as secondary info
    display_df = pd.DataFrame({
        "Experiment": cleaned["model_name"],
        "Model ID": cleaned["model_id"],
        "Accuracy (%)": cleaned["accuracy"],
        "Precision (%)": cleaned["precision"],
        "Recall (%)": cleaned["recall"],
        "F1-Score (%)": cleaned["f1_score"],
    })

    # Melt for grouped bar chart
    melted = cleaned.melt(
        id_vars=["model_name", "short_name", "model_id"],
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score (%)",
    )
    metric_title_map = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
    }
    melted["Metric"] = melted["Metric"].map(metric_title_map)

    return {
        "is_valid": True,
        "error": None,
        "display_df": display_df,
        "melted_df": melted,
        "best_model": best_model,
    }


def check_confusion_matrix_status(reports_dir: Path) -> Dict[str, Optional[Path]]:
    """Checks the availability of generated confusion matrix plot files.

    Args:
        reports_dir: Path to reports directory.

    Returns:
        Dict[str, Optional[Path]]: Mapping of model_id to existing file Path or None.
    """
    status: Dict[str, Optional[Path]] = {}
    for mid, filename in CONFUSION_FILES.items():
        p = reports_dir / filename
        status[mid] = p if p.exists() else None
    return status


# --------------------------------------------------
# Custom Styling (Dark Academic Theme)
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #f8fafc;
    }
    .hero {
        padding: 1.2rem 0 0.8rem 0;
    }
    .hero h1 {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
    }
    .result-card {
        padding: 1.6rem;
        border-radius: 14px;
        background: #1c2330;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .result-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .result-value {
        font-size: 2.4rem;
        font-weight: 900;
        margin-top: 0.4rem;
    }
    .positive-text {
        color: #34d399;
    }
    .negative-text {
        color: #f87171;
    }
    .best-banner {
        background: linear-gradient(90deg, #1e293b 0%, #1e3a5f 100%);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin: 1rem 0;
    }
    .best-banner-title {
        color: #38bdf8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }
    .best-banner-desc {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    div[data-testid="stMetric"] {
        background: #1c2330;
        border: 1px solid #334155;
        padding: 0.9rem 1.1rem;
        border-radius: 12px;
    }
    .sample-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: #334155;
        color: #e2e8f0;
        border-radius: 6px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session State Initialization
# --------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if "last_analyzed_hash" not in st.session_state:
    st.session_state.last_analyzed_hash = ""

# --------------------------------------------------
# Cached Data Loading
# --------------------------------------------------

@st.cache_data
def load_evaluation_results() -> Optional[pd.DataFrame]:
    """Loads evaluation results CSV file from reports directory."""
    csv_path = REPORTS_DIR / "evaluation_results.csv"
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def get_available_models() -> List[str]:
    """Discovers available serialized models on disk."""
    return list_available_models(models_dir=str(MODELS_DIR))


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Movie Review Sentiment Analyzer</h1>
        <p>
            An end-to-end NLP framework comparing standard baseline TF-IDF against
            context-aware (negation & contrast) feature engineering on the IMDb 50K benchmark.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# --------------------------------------------------
# Sidebar Navigation & Model Health
# --------------------------------------------------

with st.sidebar:
    st.title("🎥 Sentiment Lab")
    st.caption("Context-Aware NLP Benchmarks | IMDb 50K")
    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        "Choose a section",
        [
            "Sentiment Analyzer",
            "Model Comparison",
            "Methodology",
            "About Project",
        ],
        label_visibility="collapsed",
    )
    st.divider()

    available_models = get_available_models()
    st.subheader("Model Status")

    for mid, info in MODEL_INFO.items():
        if mid in available_models:
            st.success(f"{info['short_name']}", icon="✅")
        else:
            st.error(f"{info['short_name']}", icon="❌")

    st.caption(f"**{len(available_models)} / 4** trained models ready in `models/`")
    st.divider()

    st.caption("Academic NLP Project | Group Submission")

# ==================================================
# Page 1: Sentiment Analyzer
# ==================================================

if page == "Sentiment Analyzer":
    st.subheader("Interactive Sentiment Analysis")
    st.write(
        "Enter any movie review below or choose a curated sample to examine how different "
        "baseline and context-aware models classify sentiment."
    )

    if not available_models:
        st.error(
            "⚠️ No trained model artifacts were found in `models/`. "
            "Please run `python -m src.train` in your terminal first."
        )
        st.stop()

    # Model selector with full descriptive names
    model_choices = {
        MODEL_INFO[mid]["name"]: mid
        for mid in available_models
    }
    default_index = (
        list(model_choices.keys()).index(MODEL_INFO["baseline_logistic"]["name"])
        if MODEL_INFO["baseline_logistic"]["name"] in model_choices
        else 0
    )

    col_select, col_info = st.columns([3, 2])
    with col_select:
        selected_display = st.selectbox(
            "Select Experiment Model",
            options=list(model_choices.keys()),
            index=default_index,
            help="Choose between baseline and context-aware Naive Bayes or Logistic Regression models.",
        )
        selected_model_id = model_choices[selected_display]

    with col_info:
        st.markdown(f"**Model ID**: `{selected_model_id}`")
        st.caption(MODEL_INFO[selected_model_id]["description"])

    # Sample review loader pills
    st.markdown("**Load Curated Test Cases:**")
    sample_cols = st.columns(len(SAMPLE_REVIEWS))
    for idx, (label, text) in enumerate(SAMPLE_REVIEWS.items()):
        with sample_cols[idx]:
            if st.button(label, key=f"btn_sample_{idx}", use_container_width=True):
                st.session_state.review_input = text

    review_text = st.text_area(
        "Movie Review Text",
        value=st.session_state.review_input,
        height=170,
        max_chars=10000,
        placeholder="Type or paste a movie review here (e.g., 'The visuals were stunning, but the story was not good.')...",
        help="Maximum input length is 10,000 characters.",
    )

    # Word and character counter
    word_count = len(review_text.split()) if review_text else 0
    char_count = len(review_text) if review_text else 0
    st.caption(f"📊 Words: **{word_count}** | Characters: **{char_count:,}** / 10,000")

    btn_col, note_col = st.columns([1, 4])
    with btn_col:
        analyze_clicked = st.button("Analyze Review", type="primary", use_container_width=True)
    with note_col:
        st.caption("Inference uses the serialized end-to-end scikit-learn pipeline.")

    if analyze_clicked:
        is_valid, status_code, validation_msg = validate_review_input(review_text)

        if not is_valid:
            st.warning(validation_msg)
        else:
            if status_code == "short" and validation_msg:
                st.warning(validation_msg)

            try:
                with st.spinner("Classifying review with selected model..."):
                    pred_res = predict_sentiment(
                        review=review_text,
                        model_name=selected_model_id,
                        models_dir=str(MODELS_DIR),
                    )

                pred_sentiment = str(pred_res["sentiment"]).lower()
                pred_conf = float(pred_res["confidence"])
                full_model_name = MODEL_INFO[selected_model_id]["name"]

                # Append to history avoiding rerun duplicate
                rec = format_prediction_record(
                    review=review_text,
                    model_id=selected_model_id,
                    model_name=full_model_name,
                    sentiment=pred_sentiment,
                    confidence=pred_conf,
                )
                st.session_state.prediction_history = add_prediction_to_history(
                    st.session_state.prediction_history, rec
                )

                st.divider()
                st.subheader("Prediction Output")

                res_card_col, conf_card_col = st.columns([1, 1])

                with res_card_col:
                    if pred_sentiment == "positive":
                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="result-label">Predicted Sentiment</div>
                                <div class="result-value positive-text">POSITIVE 😊</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="result-label">Predicted Sentiment</div>
                                <div class="result-value negative-text">NEGATIVE ☹️</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with conf_card_col:
                    st.metric("Estimated Class Confidence", f"{pred_conf * 100:.2f}%")
                    st.progress(max(0.0, min(pred_conf, 1.0)))
                    st.caption(
                        "ℹ️ **Confidence Notice**: Confidence reflects the model's posterior probability "
                        "for the predicted class. It is an internal probability estimate and is not a "
                        "guaranteed measure of correctness or calibrated accuracy."
                    )

                st.info(f"**Experiment Model**: {full_model_name} (`{selected_model_id}`)")

                with st.expander("Inspect Raw Input Text"):
                    st.write(review_text)

            except FileNotFoundError as exc:
                st.error(f"Model artifact missing on disk: {exc}")
            except (TypeError, ValueError) as exc:
                st.warning(f"Validation error: {exc}")
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

    # --------------------------------------------------
    # Prediction History & CSV Export Section
    # --------------------------------------------------
    st.divider()
    st.subheader("Session Prediction History")
    st.write(
        "Track reviews analyzed during your current browser session. "
        "Data is kept in-memory and can be exported as a CSV report."
    )

    history_records = st.session_state.prediction_history

    if history_records:
        history_df = pd.DataFrame(history_records)

        table_display = history_df[[
            "timestamp",
            "model_name",
            "sentiment",
            "confidence_pct",
            "review_preview",
        ]].copy()
        table_display.columns = [
            "Timestamp",
            "Model",
            "Predicted Sentiment",
            "Confidence",
            "Review Preview",
        ]

        st.dataframe(
            table_display,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = export_history_to_csv(history_records)

        act_col1, act_col2, _ = st.columns([1.5, 1.5, 4])
        with act_col1:
            st.download_button(
                label="📥 Export History (CSV)",
                data=csv_data,
                file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download your session prediction logs as a CSV file.",
                use_container_width=True,
            )
        with act_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

    else:
        st.info("No predictions recorded in this session yet. Submit a review above to start your history log.")

# ==================================================
# Page 2: Model Comparison Dashboard
# ==================================================

elif page == "Model Comparison":
    st.subheader("Experimental Model Performance Dashboard")
    st.write(
        "Direct comparison of all four trained models evaluated on the shared "
        "9,917-review test partition (80:20 stratified split)."
    )

    raw_results = load_evaluation_results()
    dash_data = prepare_dashboard_data(raw_results)

    if not dash_data["is_valid"]:
        st.error(f"⚠️ {dash_data['error']}")
        st.info("Generate the evaluation results by running: `python -m src.evaluate`")
        st.stop()

    display_df = dash_data["display_df"]
    melted_df = dash_data["melted_df"]
    best_model = dash_data["best_model"]

    # Best-performing model banner
    st.markdown(
        f"""
        <div class="best-banner">
            <div class="best-banner-title">🏆 Top Performing Experiment</div>
            <div class="best-banner-desc">
                {best_model['model_name']} ({best_model['model_id']})
            </div>
            <div style="color: #cbd5e1; font-size: 0.95rem; margin-top: 0.3rem;">
                Achieved top test accuracy of <strong>{best_model['accuracy']:.2f}%</strong>
                and F1-score of <strong>{best_model['f1_score']:.2f}%</strong>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top summary metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Top Accuracy", f"{best_model['accuracy']:.2f}%", help="Experiment 2 (Baseline Logistic)")
    m_col2.metric("Top F1-Score", f"{best_model['f1_score']:.2f}%", help="Experiment 2 (Baseline Logistic)")
    m_col3.metric("Evaluated Models", f"{len(display_df)} Models")
    m_col4.metric("Test Partition", "9,917 Reviews", help="Held-out stratified test set")

    st.divider()

    # Tabular metrics breakdown
    st.markdown("### Performance Metrics Table")
    st.caption("All metrics computed on the 9,917-sample held-out test set. Model IDs are shown as secondary identifiers.")

    st.dataframe(
        display_df.style.format({
            "Accuracy (%)": "{:.2f}%",
            "Precision (%)": "{:.2f}%",
            "Recall (%)": "{:.2f}%",
            "F1-Score (%)": "{:.2f}%",
        }).highlight_max(subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"], color="#1e3a5f"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Grouped Bar Chart (0-100% scale)
    st.markdown("### Grouped Performance Comparison")
    st.caption("Comparison across Accuracy, Precision, Recall, and F1-Score on a standard 0% to 100% scale.")

    fig = px.bar(
        melted_df,
        x="model_name",
        y="Score (%)",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#a78bfa"],
        labels={"model_name": "Experiment", "Score (%)": "Score (%)"},
    )
    fig.update_layout(
        yaxis_range=[0, 100],
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1c2330",
        font=dict(color="#f8fafc", family="sans-serif"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
        margin=dict(l=40, r=40, t=50, b=80),
        xaxis_tickangle=-15,
    )
    fig.update_xaxes(showgrid=False, color="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#334155", color="#cbd5e1")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Dark-Themed Confusion Matrices
    st.markdown("### Confusion Matrices")
    st.caption(
        "Generated with dark theme styling matching the UI. "
        "Axes use canonical [negative, positive] sentiment labels."
    )

    cm_status = check_confusion_matrix_status(REPORTS_DIR)
    cm_cols = st.columns(2)

    for idx, (mid, img_path) in enumerate(cm_status.items()):
        col_idx = idx % 2
        with cm_cols[col_idx]:
            st.markdown(f"**{MODEL_INFO[mid]['name']}** (`{mid}`)")
            if img_path and img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Confusion matrix image missing: `reports/{CONFUSION_FILES[mid]}`")

# ==================================================
# Page 3: Methodology
# ==================================================

elif page == "Methodology":
    st.subheader("NLP Architecture & Methodology")
    st.write(
        "Comprehensive documentation of the data cleaning, feature engineering, "
        "and classification pipelines implemented in this project."
    )

    tab_data, tab_prep, tab_features, tab_models = st.tabs([
        "1. Dataset & Split",
        "2. Text Preprocessing",
        "3. Context-Aware Features",
        "4. Model Architectures",
    ])

    with tab_data:
        st.markdown(
            """
            ### IMDb 50K Movie Reviews Dataset
            - **Raw Corpus**: 50,000 movie reviews collected from IMDb with binary sentiment labels.
            - **Deduplication**: 418 exact duplicate reviews identified and eliminated, yielding **49,582 clean unique reviews**.
            - **Class Balance**: 24,884 positive reviews (50.19%) vs. 24,698 negative reviews (49.81%) — virtually 50:50 balanced.
            - **Review Lengths**: Mean review length is **231 words** (median 173 words); maximum length is 2,470 words.
            - **Stratified Partitioning**: 80:20 stratified train/test split fixed with `random_state=42`:
              - **Train Split (`data/processed/train.csv`)**: 39,665 reviews (19,907 positive / 19,758 negative).
              - **Test Split (`data/processed/test.csv`)**: 9,917 reviews (4,977 positive / 4,940 negative).
            """
        )

    with tab_prep:
        st.markdown(
            """
            ### Preprocessing & Normalization Decisions
            1. **Space-Padded HTML Stripping**:
               Replaced `<br />`, `<div>`, etc., with single space `" "` to prevent word merging (`movie<br />was` $\rightarrow$ `movie was`).
            2. **Contraction Expansion**:
               Normalizes typographic curly apostrophes (`’`, `‘`) to `'` and expands contractions (`wasn't` $\rightarrow$ `was not`, `didn't` $\rightarrow$ `did not`, `won't` $\rightarrow$ `will not`).
               *Why*: Explicitly exposes the canonical negation token `not` before stopword filtering.
            3. **Clause Boundary & Punctuation Preservation**:
               Retains delimiters (`.`, `!`, `?`, `,`, `;`, `:`) as discrete tokens to delimit clauses and negation scopes.
            4. **Negation & Contrast-Preserving Stopword Filtering**:
               Standard NLTK English stopwords naively remove `not`, `no`, `never`, which **inverts** review sentiment.
               Our custom filter strictly protects `NEGATION_WORDS` and `CONTRAST_WORDS` (`but`, `however`, `although`).
            5. **POS-Guided WordNet Lemmatization**:
               Maps Penn Treebank POS tags to WordNet constants to correctly lemmatize verbs (`running` $\rightarrow$ `run`, `was` $\rightarrow$ `be`), adjectives (`better` $\rightarrow$ `good`), and nouns (`movies` $\rightarrow$ `movie`).
            """
        )

    with tab_features:
        st.markdown(
            """
            ### Context-Aware Feature Engineering
            Standard bag-of-words ignores word order and syntactic scope. We explore two context mechanisms:

            #### 1. Negation Scope Detection (`NOT_` Prefix)
            - When a negation word (`not`, `never`, `hardly`, `without`) is encountered, all subsequent content words receive a `NOT_` prefix.
            - **Scope Termination**: The negation scope automatically terminates when a punctuation boundary (`.`, `!`, `?`, `,`, `;`) or a contrastive word is reached.
            - *Example*:
              > Raw: `"The movie was not good, but I loved it."`  
              > Marked: `"movie NOT_good , but love ."`

            #### 2. Contrastive Clause Marking (`BEFORE_BUT_` & `AFTER_BUT_`)
            - Conjunctions like `but`, `however`, `although` signal a reversal of sentiment between clauses.
            - Words preceding the conjunction receive a `BEFORE_BUT_` prefix; words following it receive an `AFTER_BUT_` prefix.
            - *Example*:
              > Raw: `"Acting was great, but screenplay was terrible."`  
              > Marked: `"BEFORE_BUT_act BEFORE_BUT_great , but AFTER_BUT_screenplay AFTER_BUT_terrible ."`
            """
        )

    with tab_models:
        st.markdown(
            """
            ### The Four Classification Experiments
            | Experiment | Feature Pipeline | Classifier | Vocabulary Size | Test Accuracy |
            | :--- | :--- | :--- | :---: | :---: |
            | **E1** | Baseline TF-IDF (Unigrams, sublinear TF) | Multinomial Naive Bayes ($\alpha=1.0$) | 46,864 | 86.13% |
            | **E2** | Baseline TF-IDF (Unigrams, sublinear TF) | Logistic Regression ($C=1.0$, L2) | 46,864 | **89.29%** |
            | **E3** | Context-Aware TF-IDF (Unigrams + Bigrams) | Multinomial Naive Bayes ($\alpha=1.0$) | 102,568 | 85.80% |
            | **E4** | Context-Aware TF-IDF (Unigrams + Bigrams) | Logistic Regression ($C=1.0$, L2) | 102,568 | 88.99% |

            All models are saved as self-contained `scikit-learn` Pipeline objects containing both the feature transformers and classifiers.
            """
        )

# ==================================================
# Page 4: About Project & Honest Discussion
# ==================================================

elif page == "About Project":
    st.subheader("About This Project & Findings")

    st.markdown(
        """
        ### Context-Aware Movie Review Sentiment Analyzer
        This academic natural language processing project investigates whether rule-based context-aware
        feature engineering (negation scope and contrastive clause marking) improves traditional
        TF-IDF linear sentiment classification on long-form movie reviews.

        ---

        ### 🔬 Honest Academic Discussion of Results
        In our experiments on the held-out 9,917-review test set:
        - **E2 (Baseline Logistic Regression)** achieved the highest overall accuracy at **89.29%** and F1-score of **89.45%**.
        - **E4 (Context-Aware Logistic Regression)** achieved **88.99%** accuracy and **89.16%** F1-score.
        - **E1 (Baseline Naive Bayes)** achieved **86.13%** accuracy vs. **E3 (Context Naive Bayes)** at **85.80%**.

        #### Key Research Insights:
        1. **Feature Sparsity & Dimensionality Expansion**:
           Marking negation (`NOT_good`) and contrastive clauses (`BEFORE_BUT_great`) expanded the vocabulary
           from **46,864 features to 102,568 features** (a 118% increase). Because linear classifiers like Naive Bayes
           and Logistic Regression rely on frequency counts rather than dense semantic embeddings, feature splitting
           dilutes token frequency and increases sparsity.
        2. **Local Sentiment vs. Document-Level Sentiment**:
           On long-form reviews averaging 231 words, overall document polarity is often governed by global word frequencies
           rather than isolated local negation flips.
        3. **Targeted Benefit vs. Overall Benchmark**:
           While context-aware features successfully prevent sentiment inversion on isolated negated clauses (e.g., *"not good"*),
           their application across entire long documents introduced slight noise into overall bag-of-words weights.

        > **Conclusion**: We report these empirical findings transparently. Context-aware feature prefixes provide valuable
        > explainability on clause-level sentiment shifters, while linear models require careful regularization when handling
        > expanded sparse feature spaces.
        """
    )

    st.divider()
    st.markdown("### Engineering Team & Stack")
    st.write(
        "- **Frameworks**: Python, Streamlit, Scikit-learn, NLTK, Pandas, NumPy, Plotly, Joblib, Pytest.\n"
        "- **Code Architecture**: Modular backend (`src/`), reproducible splits (`data/processed/`), "
        "and automated unit tests (`tests/`)."
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "🎬 Context-Aware Movie Review Sentiment Analyzer | Academic NLP Project | "
    "Dataset: IMDb 50K | Held-out Test Set: 9,917 Reviews"
)
