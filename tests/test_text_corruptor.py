from typing import List

import numpy as np
import pytest

from corrupted_text import text_corruptor


@pytest.mark.parametrize("word, words, expected", [
    ("hello", ["hello"], [0]),
    ("hello", ["hell", "hel"], [1, 2]),
    ("hello", ["pello"], [1]),
    ("hello", ["rrrrr"], [5]),
])
def test__levensthein_distance(word, words, expected):
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
    assert text_corruptor._split_by_whitespace(strings) == expected

def test_bad_autocompletes():
    pass


def test__get_rng():
    pass


def test__generate_corruption_types():
    pass


def test__hash_text_to_int():
    pass


def test__hash_text_to_str():
    pass


def test_print_corruptions():
    pass


def test_corruption_type():
    pass


def test_corruption_weights():
    pass


def test_text_corruptor():
    pass
