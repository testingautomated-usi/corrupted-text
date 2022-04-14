import logging
from typing import List

import numpy as np
from datasets import load_dataset

from corrupted_text import text_corruptor
from corrupted_text.text_corruptor import TextCorruptor


def test_imdb():
    """Tests the IMDB dataset, which is the go-to example for our corruptor.

    This takes quite a while to run."""

    logging.basicConfig(level=logging.INFO)
    nominal_train = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='train')['text']
    nominal_test = load_dataset('imdb', cache_dir="/expext2/deepgini/.external_datasets", split='test')['text']

    corruptor = TextCorruptor(base_dataset=nominal_test + nominal_train)

    # Run only on a part of the dataset
    nominal_test = nominal_test[:100]
    imdb_corrupted = corruptor.corrupt(nominal_test, severity=0.5, seed=1)

    assert len(nominal_test) == len(imdb_corrupted)

    hashes = [text_corruptor._hash_text_to_str(text) for text in nominal_test]
    assert hashes == _imdb_hashes()

    print("Printing Examples which are shown in the Readme.md.")
    nominal_sentence = ["The movie was terrible. The actors were great, but I did not like the ending. " \
                        "I will never watch this movie again."]
    for sev in np.arange(0, 1.1, 0.2):
        c = corruptor.corrupt(nominal_sentence, severity=sev, seed=1)
        print(c[0])


def _imdb_hashes() -> List[str]:
    """The hashes of the first 100 imdb corrupted texts, used for smoke-level regression testing.

    We can't compare against the actual sentences (copyright!) but this gives us enough
    confidence that the output of the corruptor is stable and the in-place generated
    datasets can be used for reproduction."""

    return ['4e58cf6a01d91cfa252332bb139e8806',
            'd74c65554e8e3261ce8bdabaa65ab2c0',
            'b3fe8fbceee505550edde27058c6dd50',
            '56861dc5817c16b2a9e96aab01a7985f',
            '20f9cafba66bbf67f84fc805e8396685',
            '287a9ce54d4be5c0404eaf4e3544b79f',
            'd426a33ef650e34bb0edf3b3fbf2f12b',
            'a99bb85dae0f0706b62f2e4a4730ad0a',
            '75d1edbe533718a8401b88280bf72a13',
            '2b9ee016473b61c556ab1744fd7ae25f',
            '7056c35a4e71693c9a22f3120149b81e',
            'b81a642df1f59f5fa08b6d94b7d0977a',
            'c14ccf7076d4cd7849f571782f16098b',
            '662baa7a0e8b1f74778c550987ca4dc7',
            '7c54010ef45ed7d06287c2a07d41d787',
            '76679643c26b6b4653ba3b87c864e44b',
            'd049697409a317441c5bd959e0789131',
            '8df9cd867e9eddd31dbfbcc3b87e4604',
            'bf69b2e873ec1e57e29b63b84b9660b8',
            '0fb87289704e237b6ad08d0f944e1a1b',
            '07ed0507fef7f6d0ebb0767773aaa930',
            '1703124868844634861b772f609a51ec',
            '814f4eb78d969ee98ed86f61f97f15f5',
            '37d1e345eb742ea5e0ee8a1abbef46bc',
            'c62f1a4fd48b47f7beff394ebe7d2788',
            'd5f10b31d3bf6d4d6f3da49f5b6385c5',
            '4c31196d5a5de83a3f6c458bf86d1684',
            '09b8eebd0e91a141096792cff8ac03a0',
            '487f13053a834df10529aa8d45748750',
            '6951c1628010eb94e3fa1921ff4cad9b',
            'bd14db82880f097f50637ae54ebd16e7',
            '4590c4a6f691a76a2553da0517b497d4',
            '1d69604dffc405d6b43cf16570592285',
            '4c98c82d311637020f6ce91f4101cbcf',
            'cd0e1887c3acc6782cd569d1939111fe',
            'df7bb77f20d2360e396ccbd022c97247',
            'ec2b2bedecfff04a9db85f3d5beed8a9',
            '7629dee9ffb114d62a3a4e687f2e6ebd',
            'd57c28432780afcde0366af9ee908a19',
            'c47aead7a8c6405df09169ab302e7c29',
            '9e7f9534846d0452a7b783edf94f8eff',
            'b82e966ecc02203456d89dc1b1f3dea3',
            '99b7fd4f7cbeee5f0c74af33d3b3bf21',
            'b81489968e0a1563c675681f63b43e55',
            '71e1cde2a3e7f570503781edc71b969b',
            'f5890545afbdd75376c45821db14add4',
            '5b239e25ab1714846bbc50fa14086045',
            '75e0c9f0f9ee467da1455d3dc0bc632d',
            '179fcc70ce5c6397917d69c978009cf6',
            '8a1cf9ccbc75580bec7e839f332ca47b',
            '5d11f4548343dbd8cd798df6c7e638a8',
            'dcf9ba4ed2d63b21ef000bc76e95dae3',
            '9a6c7b1e789ae93dadb4dcea709734a4',
            '0ce60f93b4572f6e05b1ecbc56b8d248',
            '3e1425aca9bfd7cecbc555ce4cace76b',
            '0ec0ee9ef004a2dcf923a587f2451b8d',
            '640944bc677fa1ba3a0ef109cdb842c7',
            '80a48485848df3f2e865fb49896a8d53',
            '551a64469b716cb70c5e1960a51a5633',
            '6ec981e22f7e34fdd725bfaed7cc8b15',
            '57621c8f306c932829f9ee3b77f96f54',
            '0ac02f208ed9221e6459b503a595a7e9',
            '99792a105796ebf73987af2a8f53e30d',
            '4a031011de6a1a0e185e4a4c27df945a',
            '29f6b0ccd95f2cc5b8385e66dd905171',
            'f533157fb96ac6aa00227ab916dfb1a1',
            '7468a72e42998e3705285eac36a4dc33',
            '9bafd33fd160d7e248d86ad9b7ca899b',
            'a3d3c2a12b5982c9391ffa2f5429c24f',
            '514994e1b4d02e2075d8b65201e0551e',
            '204da22c1304b0803bb63ca74c97a5be',
            'd108ca063e9e7746c2b2045373acfc99',
            'f747b22dc357d0e6be779fbe28f1a439',
            '28016b03cdd1b461017ea3319b426a2e',
            'ef8d195515d09281ae6503d1d345fdbd',
            '274a081082fdec479440d7524526964a',
            'e21ac36294873502ed06db1aa6175bce',
            'ee23409f8e1b4900447eade7a2b8b31b',
            'ae11f515aeb944a810bd92b30361e869',
            '75aa7909094175d52c1de59bfe9154f9',
            '392089ed61b1080cdec9094180d3f038',
            '5a36d9624b94f5cde6cd2e88d37d1c7f',
            'aa1fe157060f9b8bde21ea8fee0e6809',
            'e084afde0bb0e49a1f300551918d3fb9',
            '2f9d9d275669bc33be343fccecfef683',
            'f18b005ee35ed71be2e80f231b3ebd56',
            'c469fd981c0833c5a3b8eed86c736099',
            '09fb7e6056dace53cf07e6d460444271',
            'e901369575415833c4b3ca13316dfbb8',
            'ac9d336255e8d4a1dc216cadf494d46b',
            'cb4898ef31edaa875be0c661460129d1',
            '2531d09f777943e2db3368384a8c2b14',
            'c7388d0773c0cbd9ec0c4cf7a22408d6',
            '91e810bd359b23cdc8f841627c4a6589',
            '3d6b0e455a7cd9c853b82eda259621b5',
            'f51de5e732c4616e969568456652aece',
            'd978a878e2c89facc7bf6ba2ac6b146d',
            'e0bc6f4c422d039769c65c67a38e66f0',
            'd3f93f409ee5ff7412c91cb066ad24e2',
            'd4f06a33d62647784e1953ea7f6c0888',
            ]
