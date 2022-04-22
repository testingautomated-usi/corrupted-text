# Corrupted-Text: Realistic Out-of-Distribution Texts

![test](https://github.com/vikpe/python-package-starter/workflows/test/badge.svg?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Docstr-Coverage](https://badgen.net/badge/docstr-coverage/100%25/green?cache=30)](https://github.com/HunterMcGushion/docstr_coverage)
[![Python Version](https://img.shields.io/pypi/pyversions/corrupted-text)](https://img.shields.io/pypi/pyversions/corrupted-text)
[![PyPi Deployment](https://badgen.net/pypi/v/corrupted-text?cache=30)](https://pypi.org/project/corrupted-text/)
[![DOI](https://zenodo.org/badge/478863947.svg)](https://zenodo.org/badge/latestdoi/478863947)

A python library to generate out-of-distribution text datasets.
Specifically, the library applies **model-independent**, **commonplace corruptions** 
(not model-specific, worst-case adversarial corruptions).
We thus aim to allow benchmark-studies regarding robustness against **realistic outliers**.


## Implemented Corruptions

Most corruptions are based on a set of *common words*, to which a corruptor is fitted. These *common words* may be
domain specific and thus, the corruptor can be fitted with a *base dataset* from which the most common words are
extracted.

Then, the following corruptions are randomly applied on a per-word basis:

1. **Bad Autocorrection**
   Words are replaced with another, common word to which it has a small levenshtein distance. This mimicks wrong
   autocorrection, as for example done by "intelligent" mobile phone keyboards.
2. **Bad Autocompletion**
   Words are replaced with another, common word with the same starting letters. This mimicks wrong autocompletion. If no
   common word with at least 3 common start letters is found, a bad autocorrection is attempted instead.
3. **Bad Synonym** Words are replaced with a synonym, accoring to a naive, flat mapping extracted
   from [WordNet](https://wordnet.princeton.edu/), ignoring the context. This mimicks dictionary based translations,
   which are often wrong. This assumes that you are using an english-language dataset.
4. **Typo** A single letter is replaced with another, randomly chosen letter.

To any word, at most one corruption is applied, i.e., corruptions are not applied on top of each other.

The severity `]0, 1]` is a parameter to steer how many corruptions should be applied. It roughly corresponds to the
percentage of words that should be corrupted
(only *rougly* as not all bad autocompletion attempts are successful, and as sometimes, the bad synonyms consist of
multiple words, thus extending the number of words in the text).

Optionally, users can define weights to each corruption type, steering how often they should be applied.

## Accuracies

The following shows the accuracy of a regular, simple transformer model on the imdb sentiment classification dataset.
Clearly, the higher the chosen corruption severity, the lower the model accuracy.

| *Severity* | 0 (*) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9  | 1 (max!) |  
|------------|-------|-----|-----|-----|-----|------|----------|
| *Accuracy* | .87   | .81 | .78 | .75 | .71 | 0.66 | 0.64     |  

(*) No corruption, original test set.

## Installation

It's as simple as `pip install corrupted-text`.

You'll need python >= 3.7

## Usage

Usage is very straigthforward.
The following shows an example on how to corrupt the imdb sentiment classification dataset.

You can also run the example in colab: <a class="reference external" href="https://colab.research.google.com/github/testingautomated-usi/corrupted-text/blob/main/imdb_example.ipynb"><img alt="Run Example in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>


```python
import corrupted_text  # pip install corrupted-text
import logging 
from datasets import load_dataset # pip install datasets

# Enable Detailed Logging
logging.basicConfig(level=logging.INFO)

# Load the dataset (we use huggingface-datasets, but any list of strings is fine).
nominal_train = load_dataset("imdb", split="train")["text"]
nominal_test = load_dataset("imdb", split="test")["text"]

# Fit a corruptor (we fit on the training and test set,
#   but as this takes a while, you'd want to choose a smaller subset for larger datasets)
corruptor = corrupted_text.TextCorruptor(base_dataset=nominal_test + nominal_train,
                                         cache_dir=".mycache")

# Corrupt the test set with severity 0.5. The result is again a list of corrupted strings.
imdb_corrupted = corruptor.corrupt(nominal_test, severity=0.5, seed=1)
```

## Citation

    @inproceedings{Weiss2022SimpleTip, 
      title={Simple Techniques Work Surprisingly Well for Neural Network Test Prioritization and Active Learning (Replication Paper)},
      author={Weiss, Michael and Paolo, Tonella}, 
      booktitle={Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis},
      year={2022}
    }

## Other Corrupted Datasets

- [MNIST-C](https://github.com/google-research/mnist-c) by Mu and Gilmer
- [CIFAR-10-C](https://zenodo.org/record/2535967#.YmAC7nVBy8I) by Hendrycks and Dietterich
- [Imagenet-C](https://zenodo.org/record/2235448) by Hendrycks and Dietterich
- [Fashion-MNIST-C](https://github.com/testingautomated-usi/fashion-mnist-c) by Weiss and Tonella (i.e., same as `corrupted-text`)
