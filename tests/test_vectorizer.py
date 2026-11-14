"""Unit tests for the vectorizer module and baseline pipeline components."""

import tempfile
import joblib
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.vectorizer import (
    DEFAULT_BASELINE_TFIDF_CONFIG,
    ContextFeatureTransformer,
    TextPreprocessor,
    build_baseline_pipeline,
    build_context_pipeline,
    create_baseline_vectorizer,
    create_preprocessor,
)


class TestTextPreprocessor:
    """Tests for the scikit-learn compatible TextPreprocessor transformer."""

    def test_single_string_input(self):
        preprocessor = create_preprocessor()
        text = "This movie wasn't good, but the music was nice!"
        result = preprocessor.transform(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "not" in result[0]
        assert "good" in result[0]
        assert "music" in result[0]

    def test_list_input(self):
        preprocessor = create_preprocessor()
        texts = [
            "Great movie!",
            "Terrible acting and boring plot.",
        ]
        results = preprocessor.transform(texts)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "great" in results[0]
        assert "terrible" in results[1]

    def test_pandas_series_input(self):
        preprocessor = create_preprocessor()
        series = pd.Series(["Awesome film!", "Did not enjoy it."])
        results = preprocessor.transform(series)
        assert isinstance(results, list)
        assert len(results) == 2
        assert "awesome" in results[0]
        assert "not" in results[1]

    def test_empty_and_whitespace_input(self):
        preprocessor = create_preprocessor()
        results = preprocessor.transform(["", "   ", None])
        assert results == ["", "", ""]

    def test_fit_returns_self(self):
        preprocessor = create_preprocessor()
        returned = preprocessor.fit(["some text"])
        assert returned is preprocessor

    def test_scikit_learn_clone_and_get_params(self):
        preprocessor = create_preprocessor(n_jobs=2, preserve_negations=True)
        params = preprocessor.get_params()
        assert params["n_jobs"] == 2
        assert params["preserve_negations"] is True

        cloned = clone(preprocessor)
        assert cloned.n_jobs == 2
        assert cloned.preserve_negations is True

    def test_joblib_serialization(self):
        preprocessor = create_preprocessor(n_jobs=1)
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(preprocessor, tmp.name)
            loaded = joblib.load(tmp.name)
            assert isinstance(loaded, TextPreprocessor)
            out = loaded.transform(["A fantastic performance!"])
            assert "fantastic" in out[0]


class TestBaselineVectorizer:
    """Tests for the baseline TF-IDF vectorizer factory and configuration."""

    def test_default_config(self):
        vectorizer = create_baseline_vectorizer()
        assert vectorizer.lowercase is False
        assert vectorizer.ngram_range == (1, 1)
        assert vectorizer.min_df == 2
        assert vectorizer.max_df == 0.95
        assert vectorizer.sublinear_tf is True

    def test_override_config(self):
        vectorizer = create_baseline_vectorizer(min_df=1, ngram_range=(1, 2))
        assert vectorizer.min_df == 1
        assert vectorizer.ngram_range == (1, 2)

    def test_fit_transform_on_preprocessed_text(self):
        corpus = [
            "movie great enjoyable",
            "movie terrible bad",
            "plot interesting great",
        ]
        vectorizer = create_baseline_vectorizer(min_df=1)
        X = vectorizer.fit_transform(corpus)
        assert X.shape[0] == 3
        assert X.shape[1] > 0
        assert "great" in vectorizer.vocabulary_


class TestBaselinePipeline:
    """Tests for end-to-end baseline pipeline construction and serialization."""

    def test_pipeline_construction_and_structure(self):
        nb = MultinomialNB()
        pipe = build_baseline_pipeline(nb)
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "tfidf" in pipe.named_steps
        assert "classifier" in pipe.named_steps
        assert isinstance(pipe.named_steps["preprocessor"], TextPreprocessor)

    def test_pipeline_fit_predict_with_multinomial_nb(self):
        X_train = [
            "I loved this film! Absolutely wonderful, brilliant, and stunning.",
            "Great acting, fantastic story, totally superb.",
            "Worst movie ever made. Horrible, boring, and utterly terrible.",
            "Waste of time, completely awful and unwatchable.",
        ]
        y_train = ["positive", "positive", "negative", "negative"]

        pipe = build_baseline_pipeline(
            MultinomialNB(alpha=1.0),
            vectorizer_kwargs={"min_df": 1},
        )
        pipe.fit(X_train, y_train)

        test_reviews = [
            "Wonderful and brilliant movie!",
            "Horrible, awful, and completely unwatchable.",
        ]
        predictions = pipe.predict(test_reviews)
        assert predictions[0] == "positive"
        assert predictions[1] == "negative"

        proba = pipe.predict_proba(test_reviews)
        assert proba.shape == (2, 2)
        assert list(pipe.classes_) == ["negative", "positive"]

    def test_pipeline_serialization_and_loading(self):
        X_train = [
            "Great movie, wonderful acting!",
            "Terrible film, hated it completely.",
        ]
        y_train = ["positive", "negative"]

        pipe = build_baseline_pipeline(
            LogisticRegression(random_state=42),
            vectorizer_kwargs={"min_df": 1},
        )
        pipe.fit(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(pipe, tmp.name)
            loaded_pipe = joblib.load(tmp.name)

            pred = loaded_pipe.predict(["Great movie, loved every minute!"])
            assert pred[0] == "positive"
            proba = loaded_pipe.predict_proba(["Great movie, loved every minute!"])
            assert proba[0][1] > 0.5


class TestContextFeatureTransformer:
    """Tests for ContextFeatureTransformer scikit-learn transformer."""

    def test_single_string_input(self):
        cf = ContextFeatureTransformer()
        text = "movie not good but entertaining"
        result = cf.transform(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert "NOT_good" in result[0]
        assert "BEFORE_BUT_" in result[0]
        assert "AFTER_BUT_" in result[0]

    def test_list_and_series_input(self):
        cf = ContextFeatureTransformer()
        docs = ["movie not good", "plot was slow but ending was great"]
        res_list = cf.transform(docs)
        res_series = cf.transform(pd.Series(docs))
        assert len(res_list) == 2
        assert len(res_series) == 2
        assert res_list == res_series

    def test_empty_and_whitespace_input(self):
        cf = ContextFeatureTransformer()
        res = cf.transform(["", "   ", None])
        assert res == ["", "", ""]

    def test_fit_returns_self(self):
        cf = ContextFeatureTransformer()
        assert cf.fit(["test"]) is cf

    def test_joblib_serialization(self):
        cf = ContextFeatureTransformer()
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(cf, tmp.name)
            loaded = joblib.load(tmp.name)
            assert isinstance(loaded, ContextFeatureTransformer)
            out = loaded.transform(["not good but fun"])
            assert "NOT_good" in out[0]


class TestContextPipeline:
    """Tests for end-to-end context-aware pipeline construction and serialization."""

    def test_pipeline_construction_and_structure(self):
        pipe = build_context_pipeline(MultinomialNB())
        assert isinstance(pipe, Pipeline)
        assert "preprocessor" in pipe.named_steps
        assert "context_features" in pipe.named_steps
        assert "tfidf" in pipe.named_steps
        assert "classifier" in pipe.named_steps
        assert isinstance(pipe.named_steps["context_features"], ContextFeatureTransformer)

    def test_pipeline_fit_predict_raw_text(self):
        X_train = [
            "An amazing and fantastic movie! Truly wonderful experience.",
            "Loved the brilliant acting, awesome and wonderful film.",
            "Terrible, awful, and horribly boring disaster of a movie.",
            "Hated every minute, complete and utter waste of time.",
        ]
        y_train = ["positive", "positive", "negative", "negative"]

        pipe = build_context_pipeline(
            LogisticRegression(random_state=42),
            vectorizer_kwargs={"min_df": 1},
        )
        pipe.fit(X_train, y_train)

        test_reviews = [
            "Fantastic, brilliant, and wonderful film!",
            "Terrible, awful, and horribly boring.",
        ]
        preds = pipe.predict(test_reviews)
        assert preds[0] == "positive"
        assert preds[1] == "negative"

        probas = pipe.predict_proba(test_reviews)
        assert probas.shape == (2, 2)

    def test_context_pipeline_serialization_and_loading(self):
        X_train = [
            "Great movie, loved it!",
            "Terrible film, hated it.",
        ]
        y_train = ["positive", "negative"]

        pipe = build_context_pipeline(
            MultinomialNB(),
            vectorizer_kwargs={"min_df": 1},
        )
        pipe.fit(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            joblib.dump(pipe, tmp.name)
            loaded = joblib.load(tmp.name)
            assert isinstance(loaded, Pipeline)
            pred = loaded.predict(["Great movie, loved every minute!"])
            assert pred[0] == "positive"
