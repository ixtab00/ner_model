import model
import preprocessing
from general import CHAR_LOOKUP, CASING_LOOKUP
from json import dumps
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Nadam

data_loader = preprocessing.DataLoader(40, 50)
words, sents, labels, casing = data_loader.preprocess_dataset('./dataset/ner_dataset.csv')
max_sent_len = data_loader.max_sent_len
max_word_len = data_loader.max_word_len
num_label_types = len(data_loader.tag_to_idx.keys())


ner = model.build_model(len(CHAR_LOOKUP.keys()) + 3, len(data_loader.word_to_idx.keys()) + 4 , num_label_types + 1, max_sent_len, max_word_len, len(CASING_LOOKUP.keys())+1)

ner.compile(Nadam(0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ner.summary()
chkpt = ModelCheckpoint('model_weights_cnn_mod.h5', monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='min')

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=1, verbose=0, mode='max', baseline=None, restore_best_weights=False)

callbacks = [chkpt, early_stopping]
with open('./word_index.json', 'w') as file:
    file.write(dumps(data_loader.word_to_idx))

with open('./tag_index.json', 'w') as file:
    file.write(dumps(data_loader.tag_to_idx))

ner.fit(x=[words, sents, casing], y=labels, batch_size=32, callbacks=callbacks, epochs=1, validation_split=0.1)