"""Script used to generate the original IMDB-C, as used for our paper."""

import collections
import concurrent
import dataclasses
import enum
import hashlib
import itertools
import json
import logging
import os
import pickle
import re
import string
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict

import numpy as np
import polyleven
from datasets import load_dataset
from tqdm import tqdm

MAX_COMMON_START_FOR_AUTOCOMPLETE = 5
MIN_COMMON_START_FOR_AUTOCOMPLETE = 2

N_MOST_FREQUENT_WORDS = 4000

RECALC_LEVENSHTEIN_DISTANCES = True
RECALC_BAD_TRANSLATIONS = False

DATASET_DIR = ".gen/"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# TODO Replace with own github link (after acceptance), for security reasons and to avoid going offline
THESAURUS_DOWNLOAD = "https://raw.githubusercontent.com/zaibacu/thesaurus/master/en_thesaurus.jsonl"
THESAURUS_PATH = f"{DATASET_DIR}/en_thesaurus.jsonl"


def _levensthein_distance(word: str, words: List[str]) -> List[int]:
    """
    Calculates the Levenshtein distance between two words.
    Taken from (MIT licensed):
    https://github.com/nfmcclure/tensorflow_cookbook/blob/master/
        05_Nearest_Neighbor_Methods/03_Working_with_Text_Distances/03_text_distances.py
    :param word: the word to compare
    :param all_words: the list of all words
    :return: the Levenshtein distances to all words (including itself)
    """

    res = np.zeros(len(words))
    for i, w in enumerate(words):
        res[i] = polyleven.levenshtein(word, w)

    return res


def extract_common_words(ds: str, strings: List[str], force_recalc=False) -> List[str]:
    """Identifies the most common words in a passed list of strings."""
    # Save chosen words using pickle
    words_file = f'../../datasets/{ds}_levenstein_words.pkl'

    if os.path.exists(words_file) and not force_recalc:
        with open(words_file, 'rb') as f:
            return pickle.load(f)

    # Split on whitespaces
    logging.debug("[WORD EXTRACTION] Splitting dataset on whitespaces")
    words = _split_by_whitespace(strings)
    # Flatten samples and make lower case
    logging.debug("[WORD EXTRACTION] Flattening samples and making lower case")
    words = [w.lower() for l in words for w in l]
    # Remove words shorter than 4 characters
    logging.debug("[WORD EXTRACTION] Removing words shorter than 4 characters")
    words = [w for w in words if len(w) > 4]
    # Remove numbers
    logging.debug("[WORD EXTRACTION] Removing numbers")
    words = [w for w in words if not w.isdigit()]
    # Remove words which do not contain any letters
    logging.debug("[WORD EXTRACTION] Removing words which do not contain any letters")
    words = [w for w in words if any(c.isalpha() for c in w)]
    # Chose most frequent words
    logging.debug("[WORD EXTRACTION] Choosing most frequent words")
    chosen_words = dict(collections.Counter(words).most_common(N_MOST_FREQUENT_WORDS)).keys()
    chosen_words = list(chosen_words)
    # Sort alphabetically
    logging.debug("[WORD EXTRACTION] Sort chosen, unique words alphabetically")
    chosen_words.sort()

    logging.info("Finished extracting common words from imdb")
    with open(words_file, 'wb') as f:
        pickle.dump(chosen_words, f)

    return chosen_words


def _split_by_whitespace(strings: List[str]) -> List[List[str]]:
    """Splits a list of strings on whitespaces."""
    # using same regex as huggingface WhitespaceSplit
    # (see: https://huggingface.co/docs/tokenizers/python/latest/components.html)
    return [re.findall(r'\w+|[^\w\s]+', l) for l in strings]


def calculate_distances(ds: str, all_words: List[str], force_recalc: bool = False) -> np.ndarray:
    """Calculates the Levenshtein distances between the passed chosen words."""
    # TODO Case study specific cache
    path_on_fs = f'../../datasets/{ds}_levenstein_distances.npy'

    if os.path.exists(path_on_fs) and not force_recalc:
        return np.load(path_on_fs)

    def _run_for_word(word):
        return _levensthein_distance(word, all_words)

    # Note: Runtime could further be improved by leveraging the symmetry in the distance matrix.
    #       At the moment, every entry is calculated twice.
    with ThreadPoolExecutor() as executor:
        distances = list(tqdm(executor.map(_run_for_word, all_words),
                              total=len(all_words),
                              desc="Calculating Levenshtein distances"))

    distances = np.array(distances, dtype=np.uint8)
    np.save(path_on_fs, distances)

    return distances


def load_bad_translations():
    """Loads the bad translations. Downloads a wordnet thesaurus if needed."""
    # Download file if not exists
    if not os.path.isfile(THESAURUS_PATH):
        urllib.request.urlretrieve(THESAURUS_DOWNLOAD, THESAURUS_PATH)

    # Load thesaurus
    with open(THESAURUS_PATH) as f:
        data = [json.loads(line) for line in f]

    # Get simple bags of synonyms
    result = dict()
    for d in data:
        word = d['word']
        synonyms = d['synonyms']
        if len(synonyms) > 1:
            if word not in result:
                result[word] = set()
            result[word].update(synonyms)

    for word in result.keys():
        result[word] = list(result.get(word))

    return result


def bad_autocompletes(word: str,
                      start_bags: Dict[int, Dict[str, List[str]]],
                      common_letters: int) -> Optional[List[str]]:
    """Returns a list of words which start with the same letters as the passed word."""
    if common_letters < MIN_COMMON_START_FOR_AUTOCOMPLETE:
        # End of recursion, no common words found.
        # This will only rarely happen in sufficiently large datasets.
        return None

    common_letters = min(common_letters, len(word))

    res = start_bags[common_letters].get(word[:common_letters])
    if res is None or (len(res) == 1 and res[0] == word):
        # Gracefully handle case where no words start with the selected number of same letters
        return bad_autocompletes(word, start_bags, common_letters=common_letters - 1)
    if word in res:
        res.remove(word)
    return res


class CorruptionType(enum.Enum):
    """The four different corruption types, imitating natural corruptions."""
    TYPO = 0
    """Randomly replaced single chars. Imitates human typos."""
    SYNONYM = 1
    """Replacement with an (apparent) synonym. 
    
    As all context is ignored, this can (intentionally) lead to `wrong` replacements,
    given the content. This imitates word-by-word translations from a different language."""
    AUTOCOMPLETE = 2
    """Replacement with a word which starts with the same letters as the original word."""
    AUTOCORRECT = 3
    """Replacement with a very similar word.
    
    This imitates e.g. (failed) mobile phone input pattern recognitions."""


def _get_rng(seed):
    if seed is None:
        warnings.warn("Seed is not set. This may lead to different corruptions being generated each time.")
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    return rng


@dataclasses.dataclass
class CorruptionWeights:
    """Configuration of the weights of the different corruption types."""
    typo_weight: float = 0.1
    autocomplete_weight: float = 0.30
    autocorrect_weight: float = 0.30
    synonym_weight: float = 0.30


def _generate_corruption_types(seed: int,
                               num_words: int,
                               weights: CorruptionWeights,
                               ) -> List[CorruptionType]:
    """A list of randomly chosen corruption types"""
    weights = np.array([weights.typo_weight,
                        weights.autocomplete_weight,
                        weights.autocorrect_weight,
                        weights.synonym_weight])
    normalized_weights = weights / weights.sum()
    rng = _get_rng(seed)
    return [CorruptionType(rng.choice(4, p=normalized_weights)) for _ in range(num_words)]


def _hash_text(words: List[str]) -> int:
    digest = hashlib.md5(" ".join(words).encode('utf-8')).hexdigest()
    # hex to int
    hashed = int(digest, 16)
    # some collisions are ok, and smaller hashes are nicer to look at, hence % 1kk
    return hashed % 1000000


def _word_start_bags(words: List[str]) -> Dict[int, Dict[str, List[str]]]:
    """Returns dictionaries of bags with equally starting words for different start sizes."""

    def _group(num_start_chars: int) -> Dict[str, List[str]]:
        dict_res = dict()
        for word in words:
            if len(word) >= num_start_chars:
                start = word[:num_start_chars]
                if start not in dict_res:
                    dict_res[start] = []
                dict_res[start].append(word)
        return dict_res

    with ThreadPoolExecutor() as executor:
        start_sizes = list(range(MIN_COMMON_START_FOR_AUTOCOMPLETE, MAX_COMMON_START_FOR_AUTOCOMPLETE + 1))
        dicts = list(executor.map(_group, start_sizes))

    return {start_sizes[i]: dicts[i] for i in range(len(start_sizes))}


class TextCorruptor(object):
    """A class for corrupting arbitrary english (not just imdb) text datasets."""

    def __init__(self,
                 base_dataset: Optional[List[str]] = None,
                 ds_name: str = None,
                 ):
        if base_dataset is None:
            logging.info("Loading IMDB dataset to extract frequent words")
            # TODO Make huggingface cache dir configurable
            imdb = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='train+test')
            base_dataset = imdb['text']
            ds_name = 'imdb'
        self.common_words = extract_common_words(ds_name, base_dataset)
        self.start_bags = _word_start_bags(self.common_words)
        self.lev_dist = calculate_distances(ds_name, self.common_words)
        self.thesaurus = load_bad_translations()

    def corrupt(self,
                texts: List[str],
                severity: float = 0.5,
                seed: int = None,
                weights: Optional[CorruptionWeights] = None,
                ) -> List[str]:
        """Corrupts a text dataset."""

        if weights is None:
            weights = CorruptionWeights()

        assert 0 <= severity <= 1, "Severity must be between 0 and 1"

        def _corrupt_badge(words_badge: List[List[str]]) -> List[str]:
            """Parallelizable corruption job for a single dataset entry (text)."""
            badge_res = []
            for words in words_badge:
                new_text = []

                # Seed which is independent of the order and number of texts in dataset
                sentence_seed = _hash_text(words) + seed

                # Seed handling to make sure seed for individual word is not influenced by the severity.
                #   Hence, higher severity will lead to the same, but more corruptions.
                #   We do this by choosing a corruption type for each word,
                #   and only then deciding which of these corruptions should be applied based on severity.
                corruption_types = _generate_corruption_types(sentence_seed, len(words), weights)
                corruption_indexes = np.arange(len(words))
                _get_rng(sentence_seed).shuffle(corruption_indexes)
                corruption_indexes = corruption_indexes[:round(len(words) * severity)]

                for i, word in enumerate(words):
                    if np.sum(corruption_indexes == i) == 0 or len(word) < 2:
                        # Cases where no corruption should be applied
                        new_text.append(word)
                    else:
                        # Cases where corruption should be applied
                        corruption = corruption_types[i]
                        word_seed = sentence_seed + i
                        corrupt_word = self._corrupt_word(word, word_seed, corruption)
                        new_text.append(corrupt_word)

                badge_res.append(" ".join(new_text))
            return badge_res

        texts_as_words = _split_by_whitespace(texts)
        badges = []
        badge_start = 0
        while badge_start < len(texts_as_words):
            batch_size = min(len(texts_as_words) - badge_start, 1)
            badges.append(texts_as_words[badge_start:badge_start + batch_size])
            badge_start += batch_size

        print("Starting corruption")
        tqdm_bar = tqdm(total=len(badges))
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_corrupt_badge, badge): badge for badge in badges}
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                # print("bla")
                tqdm_bar.update(1)
            # TODO this changes order. need to fix
            #     print("badge done")
                corrupted_texts = []
        # corrupted_texts = list(itertools.chain.from_iterable(badge_res))


        # corrupted_texts = []
        # for i, text in tqdm(enumerate(badges), total=len(badges), desc="Corrupting dataset badges"):
        #     corrupted_texts = corrupted_texts + _corrupt_badge(text)
        return corrupted_texts

    @staticmethod
    def _corrupt_typo(text, seed: int) -> str:
        letter_index = seed % len(text)
        candidate_letters = string.ascii_lowercase.replace(text[letter_index], "")
        # MD4 hash is faster than RNG and random enough
        random_candidate_index = _hash_text([text, str(seed)]) % len(candidate_letters)
        typo = candidate_letters[random_candidate_index]
        return text[:letter_index] + typo + text[letter_index + 1:]

    def _corrupt_autocomplete(self, word, seed: int) -> str:
        candidates = bad_autocompletes(word, self.start_bags, common_letters=5)
        if candidates is None or len(candidates) == 0:
            # Edge case in small datasets, where no words start with the same letters as the word
            return self._corrupt_autocorrect(word, seed)
        # MD4 hash is faster than RNG and random enough
        random_candidate_index = _hash_text([word, str(seed)]) % len(candidates)
        return candidates[random_candidate_index]

    def _corrupt_autocorrect(self, word, seed: int) -> str:
        if word not in self.common_words:
            return word
        word_index = self.common_words.index(word)
        # Choose amongst the five most similar words, normalizing by the distance
        candidate_indices = np.argsort(self.lev_dist[word_index])[1:6]
        candidate_distances = 1 / self.lev_dist[word_index][candidate_indices]
        rng = _get_rng(seed)
        chosen_index = rng.choice(candidate_indices, p=candidate_distances / candidate_distances.sum())
        return self.common_words[chosen_index]

    def _corrupt_synonym(self, word, seed: int) -> str:
        try:
            synonyms = self.thesaurus[word]
            if len(synonyms) == 0:
                raise KeyError
        except KeyError:
            # No synonyms found, so just return a typo instead
            return self._corrupt_typo(word, seed)

        # MD4 hash is faster than RNG and random enough
        method_salt = "_corrupt_synonym"
        random_candidate_index = _hash_text([word, str(seed), method_salt]) % len(synonyms)
        # Choose a synonym at random
        return synonyms[random_candidate_index]

    def _corrupt_word(self, w, seed, corruption_type: CorruptionType) -> str:
        if corruption_type == CorruptionType.TYPO:
            return self._corrupt_typo(w, seed)
        elif corruption_type == CorruptionType.AUTOCOMPLETE:
            return self._corrupt_autocomplete(w, seed)
        elif corruption_type == CorruptionType.AUTOCORRECT:
            return self._corrupt_autocorrect(w, seed)
        elif corruption_type == CorruptionType.SYNONYM:
            return self._corrupt_synonym(w, seed)
        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")


def _print_corruptions(text: str, seed: int):
    """Visualizes corruptions by printing the passed text at various corruption levels."""
    all_texts = dict()
    for severity in np.arange(0, 1.2, 0.2):
        all_texts[severity] = corruptor.corrupt([text], seed=seed, severity=severity)

    print(f"0& {text}\\\\")
    for severity, corrupted in all_texts.items():
        print(f"{round(severity, 1)}& {corrupted[0]}\\\\")
    print()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    corruptor = TextCorruptor()

    # _print_corruptions(
    #     "The A.M. Turing Award, the ACM's most prestigious technical award, is given for major contributions of lasting importance to computing."
    #     , seed=4)

    imdb_text = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='test')['text']
    imdb_text = imdb_text[:5000]
    imdb_corrupted = corruptor.corrupt(
        imdb_text, severity=0.5, seed=1
    )

    with open(f"{DATASET_DIR}/imdb/imdb_corrupted.txt", 'wb') as f:
        pickle.dump(imdb_corrupted, f)
