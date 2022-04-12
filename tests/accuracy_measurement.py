"""Not actually a test, but a benchmark of the accuracy of the model.

Running this script will provide an overview of the inpact on accuracy using our
corruptions on a simple model."""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow import keras

from corrupted_text.text_corruptor import TextCorruptor

VOCAB_SIZE = 2000

INPUT_MAXLEN = 100


# class TransformerBlock(keras.layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [
#                 keras.layers.Dense(ff_dim, activation="relu"),
#                 keras.layers.Dense(embed_dim),
#             ]
#         )
#         self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = keras.layers.Dropout(rate)
#         self.dropout2 = keras.layers.Dropout(rate)
#
#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output)
#
#
# class TokenAndPositionEmbedding(keras.layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super(TokenAndPositionEmbedding, self).__init__()
#         self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
#
#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions


if __name__ == "__main__":

    #
    # CREATE CORRUPTED DATASETS
    #
    train_ds = load_dataset('imdb', split='train')
    x_train, y_train = train_ds['text'], train_ds['label']

    test_ds = load_dataset('imdb', split='test')
    x_test, y_test = test_ds['text'], test_ds['label']

    corruptor = TextCorruptor(train_ds['text'])

    # In parallel we create corrupted datasets (and keep them on file system)
    with ProcessPoolExecutor(max_workers=6) as executor:
        for sev in np.arange(0.1, 1.1, 0.2):
            executor.submit(corruptor.corrupt, x_test, sev, 0)

    # Fetch cached corrputed datasets from file system
    corrupted = {0: x_test}
    for sev in np.arange(0.1, 1.1, 0.2):
        corrupted[sev] = corruptor.corrupt(x_test, severity=sev, seed=0)
    #
    # #
    # # CREATE MODEL
    # #
    # tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000)
    # tokenizer.fit_on_texts(x_train)
    #
    # x_train = tokenizer.texts_to_sequences(x_train)
    # x_test = tokenizer.texts_to_sequences(x_test)
    #
    # y_train = tf.keras.utils.to_categorical(y_train)
    #
    # embed_dim = 32  # Embedding size for each token
    # num_heads = 2  # Number of attention heads
    # ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    #
    # inputs = tf.keras.layers.Input(shape=(INPUT_MAXLEN,))
    # embedding_layer = TokenAndPositionEmbedding(INPUT_MAXLEN, VOCAB_SIZE, embed_dim)
    # x = embedding_layer(inputs)
    # transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # x = transformer_block(x)
    # x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # x = tf.keras.layers.Dense(20, activation="relu")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    # outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    #
    # model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # # As opposed to the keras tutorial, we use categorical_crossentropy,
    # #   and we run 10 instead of 2 epochs, but with early stopping.
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    #
    # model.fit(
    #     x_train, y_train, batch_size=32, epochs=20, validation_split=0.1,
    #     callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    # )
    #
    #
    # #
    # # EVALUATE MODEL
    # #
    # for sev, test_set in corrupted:
    #     x = tokenizer.texts_to_sequences(x_test)
    #     model.evaluate(x, y_test)