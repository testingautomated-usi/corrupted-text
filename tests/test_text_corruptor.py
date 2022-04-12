import shutil
from typing import List

import numpy as np
import pytest

from corrupted_text import text_corruptor
from tests import dummy_dataset


@pytest.mark.parametrize("word, words, expected", [
    ("hello", ["hello"], [0]),
    ("hello", ["hell", "hel"], [1, 2]),
    ("hello", ["pello"], [1]),
    ("hello", ["rrrrr"], [5]),
])
def test__levensthein_distance(word, words, expected):
    """Test levensthein distance."""
    # Using a library for levensthein distance,
    #   hence this is just a sanity check
    actual = text_corruptor._levensthein_distance(word, words)
    assert np.all(actual == np.array(expected))
    assert actual.shape == (len(words),)


@pytest.mark.parametrize("strings, expected", [
    (["hello world"], [["hello", "world"]]),
    (["hello world", "hello world"], [["hello", "world"], ["hello", "world"]]),
    (
            ["""this is 
        multiline """],
            [["this", "is", "multiline"]]

    )
])
def test__split_by_whitespace(strings: List[str], expected: List[List[str]]):
    """Split texts by whitespace."""
    assert text_corruptor.split_by_whitespace(strings) == expected


def test_bad_autocompletes():
    """Choice of works with the same start."""
    bags_of_words = dict()
    bags_of_words[5] = {"mywor": ["mywor2", "mywor3", "mywor4", "myword5"],
                        "other": ["otherW2", "otherW3"]}

    bags_of_words[4] = {"mywo": ["mywor2", "mywor3", "mywor4", "myword5"],
                        "othe": ["otherW2", "otherW3"]}
    bags_of_words[4]["smal"] = ["smalt", "smalp"]
    # No additional test data for 3
    bags_of_words[3] = {"myw": ["mywor2", "mywor3", "mywor4", "myword5"],
                        "oth": ["otherW2", "otherW3"],
                        "sma": ["smalt", "smalp", "sma234"]}

    # For simplicity, I do not create bags for every word (e.g. for mywor2, ...)
    # in this test

    # Exact match
    assert text_corruptor.bad_autocompletes("mywor", bags_of_words, 5) == bags_of_words[5]["mywor"]
    # Longer input
    assert text_corruptor.bad_autocompletes("myword", bags_of_words, 4) == bags_of_words[4]["mywo"]
    # Shorter input
    assert text_corruptor.bad_autocompletes("myw", bags_of_words, 4) == bags_of_words[3]["myw"]
    assert text_corruptor.bad_autocompletes("small1", bags_of_words, 4) == bags_of_words[4]["smal"]
    # Test checks in smaller if not available
    assert text_corruptor.bad_autocompletes("small1", bags_of_words, 5) == bags_of_words[4]["smal"]
    # Test input word is not in result
    assert "mywor2" not in text_corruptor.bad_autocompletes("mywor2", bags_of_words, 3)
    # Test too short, nonexisting input word (i.e., end of recursion, no match found)
    assert 2 < text_corruptor.MIN_COMMON_START_FOR_AUTOCOMPLETE, "test precondition"
    assert text_corruptor.bad_autocompletes("po", bags_of_words, 5) is None


def test__hash_text_to_int():
    """Integer representation of dataset hash."""
    # Test twice the same thing gives the same result
    hash_1 = text_corruptor._hash_text_to_int(["hello"])
    hash_2 = text_corruptor._hash_text_to_int(["hello"])
    assert hash_1 == hash_2

    # Test different text gives different results
    hash_3 = text_corruptor._hash_text_to_int(["hello", "world"])
    assert hash_1 != hash_3
    hash_4 = text_corruptor._hash_text_to_int(["world"])
    assert hash_1 != hash_4

    assert type(hash_1) == type(hash_2) == type(hash_3) == type(hash_4) == int


def test__hash_text_to_str():
    """String representation of dataset hash."""
    # Test twice the same thing gives the same result
    hash_1 = text_corruptor._hash_text_to_str(["hello"])
    hash_2 = text_corruptor._hash_text_to_str(["hello"])
    assert hash_1 == hash_2

    # Test different text gives different results
    hash_3 = text_corruptor._hash_text_to_str(["hello", "world"])
    assert hash_1 != hash_3
    hash_4 = text_corruptor._hash_text_to_str(["world"])
    assert hash_1 != hash_4

    assert type(hash_1) == type(hash_2) == type(hash_3) == type(hash_4) == str


def test_corruption_type_do_not_change():
    """Simple regression test to make sure the enum values don't change"""
    assert text_corruptor.CorruptionType.TYPO.value == 0
    assert text_corruptor.CorruptionType.SYNONYM.value == 1
    assert text_corruptor.CorruptionType.AUTOCOMPLETE.value == 2
    assert text_corruptor.CorruptionType.AUTOCORRECT.value == 3


def test_corruption_weights():
    """Simple regression test to make sure the default weights don't change"""
    default_weights = text_corruptor.CorruptionWeights()
    assert default_weights.typo_weight == 0.05
    assert default_weights.autocomplete_weight == 0.30
    assert default_weights.autocorrect_weight == 0.30
    assert default_weights.synonym_weight == 0.35


def _clean_cache(ds: List[str], dict_size: int):
    cache_dir = text_corruptor.TextCorruptor(
        base_dataset=ds,
        dictionary_size=dict_size).cache_dir
    shutil.rmtree(cache_dir)


def test_text_corruptor_is_deterministic_for_same_instance():
    _clean_cache(dummy_dataset.SMALL_TRAIN_DATA, 500)
    corruptor = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500)

    c1 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=0)
    c1_recalc = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=0, force_recalculate=True)
    assert c1 == c1_recalc, "The same seed should produce the same result"

    c3 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=1)
    assert c1 != c3, "Different seeds should produce different results"
    # Double check with force recalc, to make sure result
    c3_recalc = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=1, force_recalculate=True)
    assert c1 != c3, "Different seeds should produce different results"
    assert c3 == c3_recalc, "The same seed should produce the same result"

    # Make sure length of input set does not influence the individual entry result
    sublist_size = 10
    c4 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA[:sublist_size], severity=0.8, seed=0, force_recalculate=True)
    assert c1[:sublist_size] == c4, "The input length should not influence the result"


def test_text_corruptor_is_deterministic_for_new_instance():
    _clean_cache(dummy_dataset.SMALL_TRAIN_DATA, 500)
    corruptor = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500)

    c1 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=0)

    # Make sure re-creating the corruptor does not change anything
    corruptor = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500)
    c5 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=0)
    assert c1 == c5, "Re-creating corruptor instance (with cache) influenced output."

    # Same without cache
    shutil.rmtree(corruptor.cache_dir)
    corruptor = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500)
    c6 = corruptor.corrupt(dummy_dataset.SMALL_TEST_DATA, severity=0.8, seed=0)
    assert c1 == c6, "Re-creating corruptor instance (without cache) influenced output."


def test_common_words():
    _clean_cache(dummy_dataset.SMALL_TRAIN_DATA, 500)
    _clean_cache(dummy_dataset.SMALL_TRAIN_DATA, 100)

    c_100 = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=100)

    c_500 = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500)

    assert len(c_100.common_words) == 100
    assert len(c_500.common_words) == 500
    assert all(x in c_500.common_words for x in c_100.common_words)

    # Check determinism
    _clean_cache(dummy_dataset.SMALL_TRAIN_DATA, 100)
    c_100_new = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=100)
    assert c_100.common_words == c_100_new.common_words

    c_500_new = text_corruptor.TextCorruptor(
        base_dataset=dummy_dataset.SMALL_TRAIN_DATA,
        dictionary_size=500, cache_dir=None)
    assert c_500.common_words == c_500_new.common_words

    very_frequent = ["frequent"] * 10000
    very_rare = ["suPerbStrangWord"] * 1

    ds = dummy_dataset.SMALL_TEST_DATA + very_frequent + very_rare
    c_30_test = text_corruptor.TextCorruptor(
        base_dataset=ds, dictionary_size=30)
    assert len(c_30_test.common_words) == 30
    assert very_frequent[0] in c_30_test.common_words
    assert very_rare[0] not in c_30_test.common_words
