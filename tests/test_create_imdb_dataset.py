import logging

from datasets import load_dataset

from corrupted_text.text_corruptor import TextCorruptor


def test_imdb():
    """Tests the IMDB dataset, which is the go-to example for our corruptor"""
    logging.basicConfig(level=logging.INFO)
    nominal_train = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='train')['text']
    nominal_test = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='test')['text']
    corruptor = TextCorruptor(base_dataset=nominal_test + nominal_train)

    # Run only on a part of the dataset
    nominal_test = nominal_test[:201]
    imdb_corrupted = corruptor.corrupt(
        nominal_test, severity=0.5, seed=1
    )


    assert len(nominal_test) == len(imdb_corrupted)
