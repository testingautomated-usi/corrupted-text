# Text Corruptor: Realistic Out-of-Distribution Texts

![test](https://github.com/vikpe/python-package-starter/workflows/test/badge.svg?branch=master) [![codecov](https://codecov.io/gh/vikpe/python-package-starter/branch/master/graph/badge.svg)](https://codecov.io/gh/vikpe/python-package-starter) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Corruptions

Most corruptions are based on a set of *common words*, to which a corruptor is fitted.
These *common words* may be domain specific and thus, the corruptor can be fitted
with a *base dataset* from which the most common words are extracted.

Then, the following corruptions are randomly applied on a per-word basis:

1. **Bad Autocorrection** 
Words are replaced with another, common word to which it has a small levenshtein distance.
This mimicks wrong autocorrection, as for example done by "intelligent" mobile phone keyboards.
2. **Bad Autocompletion** 
Words are replaced with another, common word with the same starting letters.
This mimicks wrong autocompletion.
If no common word with at least 3 common start letters is found, a bad autocorrection is attempted instead.
3. **Bad Synonym** Words are replaced with a synonym, accoring to a naive, flat mapping extracted 
from [WordNet](https://wordnet.princeton.edu/), ignoring the context. 
This mimicks dictionary based translations, which are often wrong as they ignore context. 
4. **Typo** A single letter is replaced with another, randomly chosen letter.

To any word, at most one corruption is applied.
The severity (`]0, 1]`) is a parameter to steer which corruptions should be applied.
It roughly corresponds to the percentage of words that should be corrupted
(only *rougly* as not all bad autocompletion attempts are successful, and as sometimes, 
the bad synonyms consist of multiple words, thus extending the number of words in the text).

Optionally, users can define weights to each corruption type, steering how often they should be applied.

## Example 
## Accuracies

The following shows the accuracy of a regular, simple transformer model 
on the imdb sentiment classification dataset.
Clearly, the higher the chosen corruption severity, the lower the model accuracy.


| *Severity* | 0 (*) | 0.1 | 0.3 | 0.5 | 0.7 | 0.9  | 1 (max!) |  
|------------|-------|-----|-----|-----|-----|------|----------|
| *Accuracy* | .87   | .81 | .78 | .75 | .71 | 0.66 | 0.64     |  

(*) No corruption, original test set.

## Installation

It's as simple as `pip install corrupted-text`. 

You'll need python 3.6 <= x <= 3.9.



## Usage


## Other Corrupted Datasets
