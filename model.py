from keras.models import Model
from keras.layers import Input, TimeDistributed, LSTM, Bidirectional, Dense, Conv1D, MaxPooling1D, Embedding, Flatten, Concatenate, Dropout
import numpy as np

def build_model(
        char_vocab_size: int,
        word_vocab_size: int,
        num_label_types: int,
        max_sent_len: int,
        max_word_len: int,
        casing_size: int,
        char_emb_dim = 32,
        word_emb_dim = 64,
        conv_filters = 32,
        conv_kernel_size = 3,
        dropot_rate = 0.4,
        lstm_units = 256
):
    char_input = Input((max_sent_len, max_word_len,), name="char_input")
    word_input = Input((max_sent_len,), name="word_input")
    casing_input = Input((max_sent_len,), name="casing_input")
    casing = Embedding(casing_size, casing_size, weights = [np.identity(casing_size)], trainable=False)(casing_input)

    chars = TimeDistributed(Embedding(input_dim=char_vocab_size, output_dim=char_emb_dim))(char_input)
    chars = Dropout(dropot_rate)(chars)
    chars = TimeDistributed(Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        activation='tanh'
    ))(chars)
    chars = TimeDistributed(MaxPooling1D())(chars)
    chars = Dropout(dropot_rate)(chars)
    chars = TimeDistributed(Flatten())(chars)

    words = Embedding(input_dim=word_vocab_size, output_dim=word_emb_dim)(word_input)

    output = Concatenate()([words, chars, casing])

    output = Bidirectional(LSTM(lstm_units, return_sequences=True, recurrent_dropout=dropot_rate/2, dropout=dropot_rate))(output)
    output = TimeDistributed(Dense(num_label_types, activation='softmax'))(output)
    return Model(inputs=[char_input, word_input, casing_input], outputs=[output])