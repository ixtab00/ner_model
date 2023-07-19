from transformer import decoder
from transformer import encoder
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from data_loaders.mt_data_loader import MTDataLoader
from preprocessing.mt_tokenizer import MTTokenizer
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
BATCH_SIZE = 64

def build_model(
        num_layers: int,
        dim_model: int,
        num_heads: int,
        dim_hidden: int,
        source_vocab_size: int,
        target_vocab_size: int,
        dropout_rate: float
):
    enc = encoder.Encoder(
        num_layers,
        num_heads,
        dim_model,
        dim_hidden,
        source_vocab_size,
        dropout_rate
    )
    dec = decoder.Decoder(
        num_layers,
        num_heads,
        dim_model,
        dim_hidden,
        target_vocab_size,
        dropout_rate
    )
    dropout = Dropout(0.5)
    encoder_input = Input((None, ), name="english")
    enc_output = enc(encoder_input)
    decoder_input = Input((None, ), name="russian")
    x = dec(decoder_input, enc_output)
    x = dropout(x)
    output = Dense(target_vocab_size, activation="softmax")(x)
    return Model([decoder_input, encoder_input], output)

def train(model, dataset, epochs: int):
    history = model.fit(dataset, epochs=epochs)
    return history

def translate(model, tokenizer, sentence: str):
    tokenized_input = tokenizer.source_vectorizer(sentence)
    decoded = "[start]"
    for i in range(tokenizer.sentence_maxlen):
        tokenized_target_sentence = tokenizer.target_vectorizer(
            [decoded])[:, :-1]
        preds = model.predict(
            [tokenized_input, tokenized_target_sentence]
        )
        sampled_token = np.argmax(preds[0, i, :])
        sampled_token = tokenizer.detokenize(sampled_token)
        decoded += " "+sampled_token
        if sampled_token == "[end]":
            break
    return decoded

chkpt = ModelCheckpoint("model_weights.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')

early_stopping = EarlyStopping(monitor='accuracy', min_delta=0, patience=1, verbose=0, mode='max', baseline=None, restore_best_weights=False)

callbacks = [chkpt, early_stopping]

dataset = MTDataLoader().load("/home/ixtab/proj/ner/dataset/machine_translation_dataset/ted_hrlr-train.parquet")
tokenizer = MTTokenizer()
dataset = tokenizer.tokenize(dataset)

model = build_model(
    4,
    256,
    4,
    512,
    50000,
    50000,
    0.1
)
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
model.fit(dataset, epochs=5)


while True:
    en = input("En: ")
    print("Ru:", translate(model, tokenizer, en))