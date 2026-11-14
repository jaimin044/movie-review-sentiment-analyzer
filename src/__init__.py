"""Context-Aware Movie Review Sentiment Analyzer.

Modules:
- preprocessing: Text cleaning, tokenization, lemmatization, stopword removal, and data splitting.
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
)

__all__ = [
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
]

