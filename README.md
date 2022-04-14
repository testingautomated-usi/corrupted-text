# Text Corruptor: Realistic Out-of-Distribution Texts

![test](https://github.com/vikpe/python-package-starter/workflows/test/badge.svg?branch=master) [![codecov](https://codecov.io/gh/vikpe/python-package-starter/branch/master/graph/badge.svg)](https://codecov.io/gh/vikpe/python-package-starter) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Corruptions

1. Bad Autocompletion
2. Bad Autocorrection
3. Bad Synonym
4. Typo

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
