import pandas as pd
import numpy as np
from typing import List
from general import CHAR_LOOKUP
from tqdm import tqdm
from keras.utils import pad_sequences

class DataLoader:
    def __init__(self, max_word_len: int, max_sent_len: int):
        self.max_word_len = max_word_len
        self.max_sent_len = max_sent_len

    def __load_data(self, path: str):
        dataset = pd.read_csv(path, encoding='latin1')
        dataset = dataset.fillna(method='ffill')
        words = list(set(dataset["Word"].values.tolist()))
        tags = list(set(dataset["Tag"].values.tolist()))
        return dataset.groupby(
            "Sentence #"
        ).apply(lambda dataset: 
                [
                    (word, tag) for word, tag in zip(dataset["Word"].values.tolist(),
                                                     dataset["Tag"].values.tolist())
                ]
            ).tolist(), words, tags
    
    def preprocess_dataset(self, path: str):
        sentences, words, tags = self.__load_data(path)
        self.words = words
        self.tags = tags
        self.word_to_idx = {word: idx for word, idx in zip(words, range(1, len(words) + 1))}
        self.tag_to_idx = {tag: idx for tag, idx in zip(tags, range(1, len(tags) + 1))}
        self.max_sent_len = min(max([len(sentence) for sentence in sentences]), self.max_sent_len)
        self.max_word_len = min(max([len(word) for word in words]), self.max_word_len)

        labels = []
        encoded_sentences = []
        encoded_words = []
        for sentence in tqdm(sentences):
            words, tags = self.__split_tags_and_words(sentence)
            cur_labels = [self.tag_to_idx[tag] for tag in tags]
            labels.append(cur_labels)
            cur_sentences = self.__encode_sentence(sentence)
            encoded_sentences.append(cur_sentences)
            chars = []
            for word in words:
                chars.append(self.__encode_word(word))
            encoded_words.append(chars)
        
        encoded_words = pad_sequences(encoded_words, self.max_sent_len, padding='post')
        encoded_sentences = pad_sequences(encoded_sentences, self.max_sent_len, padding='post')
        labels = pad_sequences(labels, self.max_sent_len, padding='post', value=self.tag_to_idx['O'])
        return np.asarray(encoded_words, dtype='int32'), np.asarray(encoded_sentences, dtype='int32'), np.asarray(labels, dtype='int32')

    def __encode_word(self, word: str) -> List[int]:
        encoded_word = [0 for _ in range(self.max_word_len)]
        unk_idx = len(CHAR_LOOKUP.keys()) + 1
        word = word[:self.max_word_len]
        for i, char in enumerate(word):
            code = CHAR_LOOKUP.get(char, unk_idx)
            encoded_word[i] = code
        return encoded_word
    
    def __encode_sentence(self, sentence: List[str]) -> List[int]:
        encoded_sentence = [0 for _ in range(self.max_sent_len)]
        unk_idx = len(self.words) + 2
        sentence = sentence[:self.max_sent_len]   
        for i, word in enumerate(sentence):
            code = self.word_to_idx.get(word, unk_idx)
            encoded_sentence[i] = code
        return encoded_sentence
    
    def __split_tags_and_words(self, sentence: List):
        labels = []
        words = []
        for entry in sentence:
            words.append(entry[0])
            labels.append(entry[1])
        return words, labels