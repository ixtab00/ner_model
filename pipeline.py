from preprocessing import DataLoader
from keras.utils import pad_sequences
from keras.models import load_model
from typing import Tuple, Dict, List
import numpy as np
from general import CHAR_LOOKUP, CASING_LOOKUP

class EncodingStage(DataLoader):
    def __init__(self, max_word_len: int, max_sent_len: int, windex: Dict, decoder):
        super().__init__(max_word_len, max_sent_len)
        self.pass_to = []
        self.temp_data = []
        self.word_to_idx = windex
        self.link(decoder)

    def forward(self, data: str):
        data = data.split(' ')
        self.pass_data(data)
        encoded_sentences = []
        encoded_words = []
        encoded_casings = []
        cur_sentences = self.__encode_sentence(data)
        encoded_sentences.append(cur_sentences)
        chars = []
        cur_casing = self.__get_casing(data)
        encoded_casings.append(cur_casing)
        for word in data:
            chars.append(self.__encode_word(word))
        encoded_words.append(chars)
        
        encoded_words = pad_sequences(encoded_words, self.max_sent_len, padding='post')
        encoded_sentences = pad_sequences(encoded_sentences, self.max_sent_len, padding='post')
        encoded_casings = pad_sequences(encoded_casings, self.max_sent_len, padding='post')

        return encoded_words, encoded_sentences, encoded_casings
    
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
        unk_idx = len(self.word_to_idx.keys()) + 2
        sentence = sentence[:self.max_sent_len]   
        for i, word in enumerate(sentence):
            code = self.word_to_idx.get(word, unk_idx)
            encoded_sentence[i] = code
        return encoded_sentence
    
    def __get_casing(self, sentence: List[str]) -> List[int]:
        encoded_casing= [0 for _ in range(self.max_sent_len)]
        unk_idx = CASING_LOOKUP['other']
        sentence = sentence[:self.max_sent_len]
        for i, word in enumerate(sentence):
            casing = ''
            for char in word:
                if char.isdigit():
                    casing = 'numeric'
            if word[0].isupper():
                casing = 'initial_upper'
            elif word.islower():
                casing = 'all_lower'
            elif word.isupper():
                casing = 'all_upper'
            encoded_casing[i] = CASING_LOOKUP.get(casing, unk_idx)
        return encoded_casing
    
    
    def link(self, stage):
        self.pass_to.append(stage)

    def pass_data(self, data):
        for stage in self.pass_to:
            stage.temp_data.append(data)
    
    def clear_state(self):
        self.temp_data.clear()
    

class ComputingStage:
    def __init__(self, path: str):
        self.model = load_model(path)
        self.model.summary()
    
    def forward(self, data: Tuple):
        encoded_words, encoded_sentences, encoded_casings = data
        preds = self.model.predict([encoded_words, encoded_sentences, encoded_casings], verbose = 0)
        return preds
    

class DecodingStage:
    def __init__(self, tags_encoding: Dict):
        self.index_to_tag = {index: tag for tag, index in tags_encoding.items()}
        self.pass_to = []
        self.temp_data = []
    
    def forward(self, data):
        preds = np.argmax(data, axis=-1)
        preds = np.reshape(preds, (50))
        decoded_preds = []
        for pred in preds:
            decoded_preds.append(self.index_to_tag[pred])
        output = [(self.temp_data[0][i], decoded_preds[i]) for i in range(len(self.temp_data[0]))]
        self.temp_data.clear()
        return output
    
    def link(self, stage):
        self.pass_to.append(stage)

    def pass_data(self, data):
        for stage in self.pass_to:
            stage.temp_data.append(data)
    
    def clear_state(self):
        self.temp_data.clear()


class Pipeline():
    def __init__(self, *stages):
        self.stages = stages

    def pipeline(self, data: str, mode = "default"):
        if mode == "default":
            for stage in self.stages:
                data = stage.forward(data)
            
        return data
