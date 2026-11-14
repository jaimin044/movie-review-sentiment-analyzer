from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import (
    predict_sentiment,
    list_available_models,
)

# --------------------------------------------------
# Configuration
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

MODEL_INFO = {
    "baseline_nb": {
        "name": "E1 - Baseline Naive Bayes",
        "description": "Unigram TF-IDF with Multinomial Naive Bayes.",
        "family": "Baseline",
    },
    "baseline_logistic": {
        "name": "E2 - Baseline Logistic Regression",
        "description": "Unigram TF-IDF with Logistic Regression.",
        "family": "Baseline",
    },
    "context_nb": {
        "name": "E3 - Context-Aware Naive Bayes",
        "description": "Context-aware TF-IDF with Multinomial Naive Bayes.",
        "family": "Context-Aware",
    },
    "context_logistic": {
        "name": "E4 - Context-Aware Logistic Regression",
        "description": "Context-aware TF-IDF with Logistic Regression.",
        "family": "Context-Aware",
    },
}

CONFUSION_FILES = {
    "baseline_nb": "confusion_matrix_baseline_nb.png",
    "baseline_logistic": "confusion_matrix_baseline_logistic.png",
    "context_nb": "confusion_matrix_context_nb.png",
    "context_logistic": "confusion_matrix_context_logistic.png",
}


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #aab4c3;
        font-size: 1.05rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 14px;
        background: #1c2330;
        border: 1px solid #354052;
        text-align: center;
    }

    .result-label {
        font-size: 0.9rem;
        color: #aab4c3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }

    .positive {
        color: #36d399;
    }

    .negative {
        color: #ff6b81;
    }

    div[data-testid="stMetric"] {
        background: #1c2330;
        border: 1px solid #354052;
        padding: 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Data loading
# --------------------------------------------------

@st.cache_data
def load_evaluation_results():
    csv_path = REPORTS_DIR / "evaluation_results.csv"

    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


def get_available_models():
    return list_available_models(models_dir=str(MODELS_DIR))


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎬 Movie Review Sentiment Analyzer</h1>
        <p>
            An experimental NLP application comparing baseline and
            context-aware machine learning models for movie review sentiment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🎥 Sentiment Lab")

    st.caption("IMDb 50K Movie Review Classification")

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

    st.subheader("Model Availability")

    for model_id, info in MODEL_INFO.items():
        if model_id in available_models:
            st.success(info["name"], icon="✅")
        else:
            st.error(info["name"], icon="❌")
    st.caption(f"{len(available_models)}/4 trained models available")


# --------------------------------------------------
# Sentiment Analyzer
# --------------------------------------------------

if page == "Sentiment Analyzer":

    st.subheader("Analyze a Movie Review")

    st.write(
        "Enter a movie review and select one of the four trained "
        "experiments to predict its sentiment."
    )

    if not available_models:
        st.error(
            "No trained model artifacts were found in the models directory. "
            "Run `python -m src.train` first."
        )
        st.stop()

    model_options = {
        MODEL_INFO[mid]["name"]: mid
        for mid in available_models
    }

    selected_label = st.selectbox(
        "Select experiment",
        list(model_options.keys()),
        index=(
            list(model_options.keys()).index(
                "E2 - Baseline Logistic Regression"
            )
            if "E2 - Baseline Logistic Regression" in model_options
            else 0
        ),
    )

    selected_model = model_options[selected_label]

    st.caption(MODEL_INFO[selected_model]["description"])

    review = st.text_area(
        "Your movie review",
        height=190,
        max_chars=10000,
        placeholder=(
            "Example: The acting was brilliant, but the story "
            "was painfully boring..."
        ),
    )

    col1, col2 = st.columns([1, 5])

    with col1:
        predict_clicked = st.button(
            "Analyze Review",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.caption(f"Selected model: {selected_model}")

    if predict_clicked:

        if not review.strip():
            st.warning("Please enter a movie review first.")

        else:
            try:
                with st.spinner("Analyzing your review..."):

                    result = predict_sentiment(
                        review=review,
                        model_name=selected_model,
                        models_dir=str(MODELS_DIR),
                    )

                sentiment = result["sentiment"].lower()
                confidence = result["confidence"]

                st.divider()
                st.subheader("Prediction Result")

                result_col, confidence_col = st.columns(2)

                with result_col:
                    if sentiment == "positive":
                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="result-label">
                                    Predicted Sentiment
                                </div>
                                <div class="result-value positive">
                                    POSITIVE 😊
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    else:
                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="result-label">
                                    Predicted Sentiment
                                </div>
                                <div class="result-value negative">
                                    NEGATIVE ☹️
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with confidence_col:
                    st.metric(
                        "Model confidence",
                        f"{confidence * 100:.2f}%",
                    )

                    st.progress(
                        max(0.0, min(float(confidence), 1.0))
                    )

                st.caption(
                    "Confidence is the model's estimated probability "
                    "for its predicted class. It is not necessarily "
                    "a calibrated probability of correctness."
                )

                st.info(f"Model used: {MODEL_INFO[selected_model]['name']}")

                with st.expander("View submitted review"):
                    st.write(review)

            except FileNotFoundError as exc:
                st.error(f"Model artifact missing: {exc}")

            except (TypeError, ValueError) as exc:
                st.warning(str(exc))

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# --------------------------------------------------
# Model Comparison
# --------------------------------------------------

elif page == "Model Comparison":

    st.subheader("Experimental Model Comparison")

    results = load_evaluation_results()

    if results is None:
        st.error(
            "Evaluation results not found. Run "
            "`python -m src.evaluate` first."
        )
        st.stop()

    if results.empty:
        st.warning("The evaluation results file is empty.")
        st.stop()

    required_columns = {
        "model_id",
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        st.error(
            f"Evaluation CSV is missing columns: {sorted(missing_columns)}"
        )
        st.stop()

    results = results.copy()

    # Convert metric values into percentages for display.
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    for column in metric_columns:
        results[column] = results[column] * 100

    best_model = results.loc[results["accuracy"].idxmax()]

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Best Accuracy",
        f"{best_model['accuracy']:.2f}%",
    )

    metric2.metric(
        "Experiments Evaluated",
        len(results),
    )

    metric3.metric(
        "Best Performing Model",
        best_model["model_id"],
    )

    st.divider()

    st.markdown("### Performance Metrics")

    display_columns = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    display_df = results[display_columns].copy()

    display_df.columns = [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
    ]

    st.dataframe(
        display_df.style.format(
            {
                "Accuracy (%)": "{:.2f}",
                "Precision (%)": "{:.2f}",
                "Recall (%)": "{:.2f}",
                "F1 Score (%)": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Accuracy Comparison")

    accuracy_chart = results.set_index("model_name")[["accuracy"]]

    st.bar_chart(accuracy_chart, y="accuracy")

    st.caption("Accuracy is displayed as a percentage.")

    st.markdown("### Detailed Metric Comparison")

    chart_df = results.set_index("model_name")[
        ["accuracy", "precision", "recall", "f1_score"]
    ]

    chart_df.columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    st.bar_chart(chart_df)

    st.divider()

    st.markdown("### Confusion Matrices")

    matrix_models = st.columns(2)

    for index, model_id in enumerate(CONFUSION_FILES):

        image_path = REPORTS_DIR / CONFUSION_FILES[model_id]

        with matrix_models[index % 2]:

            st.markdown(
                f"**{MODEL_INFO[model_id]['name']}**"
            )

            if image_path.exists():
                st.image(
                    str(image_path),
                    use_container_width=True,
                )
            else:
                st.warning(
                    f"Confusion matrix not found: {image_path.name}"
                )

    st.caption(
        "Confusion matrix axes use negative and positive sentiment labels."
    )


# --------------------------------------------------
# Methodology
# --------------------------------------------------

elif page == "Methodology":

    st.subheader("Project Methodology")

    st.markdown(
        """
        This project investigates binary sentiment classification
        using the IMDb 50K movie review dataset.

        ### Dataset

        - 50,000 IMDb movie reviews.
        - Positive and negative sentiment labels.
        - Duplicate and invalid records removed during preprocessing.
        - Stratified 80:20 training and testing split.
        - Random state: 42.

        ### Text Preprocessing

        The preprocessing pipeline includes:

        1. HTML tag and URL removal.
        2. Contraction expansion.
        3. Tokenization.
        4. Stopword removal while preserving negation words.
        5. POS-guided lemmatization.
        6. Punctuation and contrast-word preservation.

        ### Context-Aware Feature Engineering

        Two context-aware mechanisms are investigated.

        **Negation Scope Detection**

        Words within a negation scope are marked using the `NOT_` prefix.

        Example:

        `not good` → `NOT_good`

        **Contrastive Clause Marking**

        Contrastive conjunctions such as `but`, `however`, and `although`
        are used to distinguish clauses.

        Tokens before and after contrastive conjunctions receive
        different prefixes.

        ### Four Experiments

        | Experiment | Feature Extraction | Classifier |
        |---|---|---|
        | E1 | Baseline TF-IDF | MultinomialNB |
        | E2 | Baseline TF-IDF | Logistic Regression |
        | E3 | Context-Aware TF-IDF | MultinomialNB |
        | E4 | Context-Aware TF-IDF | Logistic Regression |

        ### Evaluation

        Models are compared using:

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion matrices
        """
    )


# --------------------------------------------------
# About Project
# --------------------------------------------------

elif page == "About Project":

    st.subheader("About This Project")

    st.markdown(
        """
        ### Context-Aware Movie Review Sentiment Analyzer

        This academic NLP project compares conventional TF-IDF
        sentiment classification with context-aware feature engineering.

        **Main objectives**

        - Build baseline sentiment classification models.
        - Investigate the importance of negation in sentiment.
        - Investigate contrastive clauses in movie reviews.
        - Compare Naive Bayes and Logistic Regression.
        - Evaluate all experiments on a common test dataset.

        **Technology Stack**

        - Python
        - Scikit-learn
        - NLTK
        - Pandas
        - NumPy
        - Streamlit
        - Matplotlib and Seaborn

        **Dataset:** IMDb 50K Movie Reviews.

        **Classification task:** Binary sentiment classification.
        """
    )

    st.divider()

    st.markdown("### Experimental Findings")

    st.write(
        "The baseline Logistic Regression model achieved the highest "
        "reported test accuracy at 89.29%. Context-aware models "
        "showed slightly lower overall accuracy in this experiment. "
        "This provides an opportunity to investigate the effects "
        "of vocabulary expansion and feature sparsity."
    )

    st.caption(
        "The displayed results are based on the saved evaluation report."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Context-Aware Movie Review Sentiment Analyzer | "
    "IMDb 50K | NLP Academic Project"
)
