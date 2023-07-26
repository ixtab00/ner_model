import model
import preprocessing
from general import CHAR_LOOKUP

data_loader = preprocessing.DataLoader(30, 50)
words, sents, labels = data_loader.preprocess_dataset('./dataset/ner_dataset.csv')
max_sent_len = data_loader.max_sent_len
max_word_len = data_loader.max_word_len
num_label_types = len(data_loader.tag_to_idx.keys())


ner = model.build_model(len(CHAR_LOOKUP.keys()) + 3, len(data_loader.word_to_idx.keys()) + 4 , num_label_types + 1, max_sent_len, max_word_len)

ner.compile('nadam', loss='sparse_categorical_crossentropy')
ner.summary()

ner.fit(x=[words, sents], y=labels, batch_size=32)