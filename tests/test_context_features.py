from src.context_features import (
    apply_context_features,
    mark_contrastive_clauses,
    mark_negation_scope,
)


def test_negation_scope_stops_at_punctuation():
    assert mark_negation_scope("The movie is not good. The ending is interesting.") == (
        "the movie is not NOT_good the ending is interesting"
    )


def test_negation_scope_stops_at_clause_boundary():
    assert mark_negation_scope("It is not good but entertaining") == (
        "it is not NOT_good but entertaining"
    )


def test_contrastive_clauses_are_positionally_marked():
    assert mark_contrastive_clauses("The story is slow but the ending is great") == (
        "BEFORE_BUT_the BEFORE_BUT_story BEFORE_BUT_is BEFORE_BUT_slow "
        "but AFTER_BUT_the AFTER_BUT_ending AFTER_BUT_is AFTER_BUT_great"
    )


def test_context_features_combine_negation_and_contrast():
    result = apply_context_features("The movie is not good but entertaining.")
    assert result == (
        "BEFORE_BUT_the BEFORE_BUT_movie BEFORE_BUT_is BEFORE_BUT_not "
        "BEFORE_BUT_NOT_good but AFTER_BUT_entertaining"
    )
