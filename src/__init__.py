"""Context-Aware Movie Review Sentiment Analyzer.

Modules:
- preprocessing: Text cleaning, tokenization, lemmatization, stopword removal, and data splitting.
- vectorizer: TextPreprocessor transformer, TF-IDF vectorizer, and baseline pipeline construction.
- train: Model training, validation, and pipeline serialization.
- evaluate: Model evaluation, metrics computation, reports, and visualization.
- predict: Inference pipeline, input validation, and sentiment prediction.
"""

from src.preprocessing import (
    clean_text,
    tokenize_text,
    lemmatize_tokens,
    preprocess_text,
    remove_stopwords,
    expand_contractions,
    load_dataset,
    clean_dataset,
    split_data,
    prepare_and_save_splits,
    run_preprocessing,
    NEGATION_WORDS,
    CONTRACTION_MAP,
    CONTRAST_WORDS,
    REQUIRED_COLUMNS,
)
from src.vectorizer import (
    DEFAULT_BASELINE_TFIDF_CONFIG,
    TextPreprocessor,
    build_baseline_pipeline,
    create_baseline_vectorizer,
    create_preprocessor,
    build_context_pipeline,
    ContextFeatureTransformer,
)
from src.train import (
    LABEL_ORDER,
    load_and_validate_data,
    train_baseline_models,
    train_context_models,
    validate_dataset_schema,
)
from src.evaluate import (
    MODEL_REGISTRY,
    compute_metrics,
    evaluate_all,
    load_saved_pipelines,
    load_test_data,
    plot_confusion_matrix,
    plot_model_comparison,
)
from src.predict import (
    MODEL_FILES,
    clear_model_cache,
    get_supported_models,
    list_available_models,
    load_pipeline,
    predict_sentiment,
    predict_sentiment_batch,
    validate_review_input,
)

__all__ = [
    # Preprocessing
    "clean_text",
    "tokenize_text",
    "lemmatize_tokens",
    "preprocess_text",
    "remove_stopwords",
    "expand_contractions",
    "load_dataset",
    "clean_dataset",
    "split_data",
    "prepare_and_save_splits",
    "run_preprocessing",
    "NEGATION_WORDS",
    "CONTRACTION_MAP",
    "CONTRAST_WORDS",
    "REQUIRED_COLUMNS",
    # Vectorizer
    "DEFAULT_BASELINE_TFIDF_CONFIG",
    "TextPreprocessor",
    "build_baseline_pipeline",
    "create_baseline_vectorizer",
    "create_preprocessor",
    "build_context_pipeline",
    "ContextFeatureTransformer",
    # Train
    "LABEL_ORDER",
    "load_and_validate_data",
    "train_baseline_models",
    "train_context_models",
    "validate_dataset_schema",
    # Evaluate
    "MODEL_REGISTRY",
    "compute_metrics",
    "evaluate_all",
    "load_saved_pipelines",
    "load_test_data",
    "plot_confusion_matrix",
    "plot_model_comparison",
    # Predict
    "MODEL_FILES",
    "clear_model_cache",
    "get_supported_models",
    "list_available_models",
    "load_pipeline",
    "predict_sentiment",
    "predict_sentiment_batch",
    "validate_review_input",
]
