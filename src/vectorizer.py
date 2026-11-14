"""Vectorizer module for Context-Aware Movie Review Sentiment Analyzer.

Provides scikit-learn compatible text preprocessing transformers, baseline
TF-IDF vectorizer construction, and pipeline assembly utilities.
Ensures identical preprocessing behavior between training and inference pipelines.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocess_text

# Default baseline TF-IDF vectorizer configuration
# Strictly unigram, no double-lowercasing, min_df=2, max_df=0.95, sublinear_tf=True
DEFAULT_BASELINE_TFIDF_CONFIG: dict[str, Any] = {
    "lowercase": False,  # Preprocessing already performs lowercasing
    "ngram_range": (1, 1),  # Baseline strictly uses unigrams
    "min_df": 2,  # Ignore words appearing in fewer than 2 documents
    "max_df": 0.95,  # Ignore words appearing in >95% of documents
    "sublinear_tf": True,  # Apply sublinear tf scaling (1 + log(tf))
}


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for text preprocessing.

    Wraps the modular `preprocess_text` function from `src.preprocessing`
    to allow end-to-end integration within a scikit-learn Pipeline.
    Supports single-string, list-of-strings, and pandas Series inputs,
    and enables multi-worker batch processing via joblib.
    """

    def __init__(
        self,
        lowercase: bool = True,
        expand_contractions_flag: bool = True,
        remove_stopwords_flag: bool = True,
        preserve_negations: bool = True,
        preserve_contrast: bool = True,
        preserve_punctuation: bool = True,
        use_pos_lemmatization: bool = True,
        n_jobs: int = 1,
    ) -> None:
        """Initializes the TextPreprocessor transformer.

        Args:
            lowercase: Whether to convert text to lowercase. Defaults to True.
            expand_contractions_flag: Whether to expand contractions. Defaults to True.
            remove_stopwords_flag: Whether to remove stopwords. Defaults to True.
            preserve_negations: Whether to protect negation words. Defaults to True.
            preserve_contrast: Whether to protect contrastive conjunctions. Defaults to True.
            preserve_punctuation: Whether to preserve clause/sentence punctuation. Defaults to True.
            use_pos_lemmatization: Whether to use POS tags for lemmatization. Defaults to True.
            n_jobs: Number of parallel jobs for batch transformation. Defaults to 1.
        """
        self.lowercase = lowercase
        self.expand_contractions_flag = expand_contractions_flag
        self.remove_stopwords_flag = remove_stopwords_flag
        self.preserve_negations = preserve_negations
        self.preserve_contrast = preserve_contrast
        self.preserve_punctuation = preserve_punctuation
        self.use_pos_lemmatization = use_pos_lemmatization
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any = None) -> "TextPreprocessor":
        """Fits the transformer on input data.

        Stateless transformer; returns self.

        Args:
            X: Input documents (ignored during fit).
            y: Target labels (ignored during fit).

        Returns:
            TextPreprocessor: Self instance.
        """
        return self

    def _preprocess_single(self, text: str) -> str:
        """Preprocesses a single text document string."""
        if not isinstance(text, str) or not text.strip():
            return ""
        result = preprocess_text(
            text=text,
            lowercase=self.lowercase,
            expand_contractions_flag=self.expand_contractions_flag,
            remove_stopwords_flag=self.remove_stopwords_flag,
            preserve_negations=self.preserve_negations,
            preserve_contrast=self.preserve_contrast,
            preserve_punctuation=self.preserve_punctuation,
            use_pos_lemmatization=self.use_pos_lemmatization,
            return_tokens=False,
        )
        return str(result)

    def transform(self, X: Union[str, Sequence[str], pd.Series, Iterable[str]]) -> list[str]:
        """Transforms a collection of raw text documents into preprocessed strings.

        Args:
            X: A single review string or iterable/Series/list of review strings.

        Returns:
            list[str]: List of preprocessed text strings.
        """
        if isinstance(X, str):
            return [self._preprocess_single(X)]

        if hasattr(X, "tolist"):
            docs = [str(x) if pd.notna(x) else "" for x in X.tolist()]
        elif isinstance(X, (list, tuple)):
            docs = [str(x) if x is not None else "" for x in X]
        else:
            docs = [str(x) if x is not None else "" for x in X]

        if not docs:
            return []

        if self.n_jobs != 1 and len(docs) > 50:
            return Parallel(n_jobs=self.n_jobs, batch_size=64)(
                delayed(self._preprocess_single)(doc) for doc in docs
            )
        else:
            return [self._preprocess_single(doc) for doc in docs]


def create_preprocessor(
    lowercase: bool = True,
    expand_contractions_flag: bool = True,
    remove_stopwords_flag: bool = True,
    preserve_negations: bool = True,
    preserve_contrast: bool = True,
    preserve_punctuation: bool = True,
    use_pos_lemmatization: bool = True,
    n_jobs: int = 1,
) -> TextPreprocessor:
    """Factory function to instantiate a configured TextPreprocessor transformer.

    Args:
        lowercase: Whether to convert text to lowercase. Defaults to True.
        expand_contractions_flag: Whether to expand contractions. Defaults to True.
        remove_stopwords_flag: Whether to remove stopwords. Defaults to True.
        preserve_negations: Whether to protect negation words. Defaults to True.
        preserve_contrast: Whether to protect contrastive conjunctions. Defaults to True.
        preserve_punctuation: Whether to preserve punctuation tokens. Defaults to True.
        use_pos_lemmatization: Whether to use POS tags for lemmatization. Defaults to True.
        n_jobs: Number of parallel worker jobs. Defaults to 1.

    Returns:
        TextPreprocessor: Configured scikit-learn compatible transformer.
    """
    return TextPreprocessor(
        lowercase=lowercase,
        expand_contractions_flag=expand_contractions_flag,
        remove_stopwords_flag=remove_stopwords_flag,
        preserve_negations=preserve_negations,
        preserve_contrast=preserve_contrast,
        preserve_punctuation=preserve_punctuation,
        use_pos_lemmatization=use_pos_lemmatization,
        n_jobs=n_jobs,
    )


def create_baseline_vectorizer(**kwargs: Any) -> TfidfVectorizer:
    """Factory function to instantiate the baseline TF-IDF vectorizer.

    Uses default configuration:
    - lowercase: False (handled during preprocessing)
    - ngram_range: (1, 1) (unigrams)
    - min_df: 2
    - max_df: 0.95
    - sublinear_tf: True

    Any keyword arguments override the default configuration.

    Args:
        **kwargs: Overriding parameters for TfidfVectorizer.

    Returns:
        TfidfVectorizer: Configured TF-IDF vectorizer.
    """
    config = dict(DEFAULT_BASELINE_TFIDF_CONFIG)
    config.update(kwargs)
    return TfidfVectorizer(**config)


def build_baseline_pipeline(
    classifier: Any,
    preprocessor_kwargs: Optional[dict[str, Any]] = None,
    vectorizer_kwargs: Optional[dict[str, Any]] = None,
) -> Pipeline:
    """Constructs a complete scikit-learn Pipeline with preprocessing, TF-IDF, and classifier.

    Pipeline structure:
        Raw review -> TextPreprocessor -> TfidfVectorizer -> Classifier -> Prediction

    Args:
        classifier: A scikit-learn compatible classifier (e.g., MultinomialNB or LogisticRegression).
        preprocessor_kwargs: Optional kwargs dictionary for TextPreprocessor.
        vectorizer_kwargs: Optional kwargs dictionary for TfidfVectorizer.

    Returns:
        Pipeline: Assembled, unfitted scikit-learn Pipeline.
    """
    preprocessor = create_preprocessor(**(preprocessor_kwargs or {}))
    vectorizer = create_baseline_vectorizer(**(vectorizer_kwargs or {}))
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
