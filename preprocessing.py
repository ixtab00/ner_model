from typing import List, Dict
from keras.preprocessing.sequence import pad_sequences

class Preprocessor():
    def __init__(self, sentence_maxlen: int):
        self.sentnce_maxlen = sentence_maxlen

    def __get_encodings(self, unique_words: List, unique_tags: List):
        unique_words.append("PAD")
        unique_words.append("UNK")
        words_encoding = {word: index for word, index in zip(unique_words, range(len(unique_words)))}
        tags_encoding = {tag: index for tag, index in zip(unique_tags, range(len(unique_tags)))}

        return words_encoding, tags_encoding
    
    def __to_sentences(self, sentences: List, words_encoding: Dict):
        return [[words_encoding[entry[0]] for entry in sentence] for sentence in sentences]
    
    def __to_tags(self, sentences: List, tags_encoding: Dict):
        return [[tags_encoding[entry[1]] for entry in sentence] for sentence in sentences]
    
    def __pad_sequence(self, sequence: List[List], value: int):
        return pad_sequences(sequences = sequence,
                             maxlen = self.sentnce_maxlen,
                             padding = 'post',
                             value = value)
    
    def process(self, sentences: List, unique_words: Dict, unique_tags: Dict):
        self.words_encoding, self.tags_encoding = self.__get_encodings(unique_words, unique_tags)
        transformed_sentences = self.__to_sentences(sentences, self.words_encoding)
        tags = self.__to_tags(sentences, self.tags_encoding)
        transformed_sentences = self.__pad_sequence(transformed_sentences, self.words_encoding["PAD"])
        tags = self.__pad_sequence(tags, self.tags_encoding["O"])

        return transformed_sentences, tags