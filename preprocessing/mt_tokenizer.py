from typing import Any
from keras.layers import TextVectorization
from preprocessing.base_tokenizer import Tokrnizer
import tensorflow as tf

class MTTokenizer(Tokrnizer):
    def __init__(self, max_vocab_size = 50000, sentence_maxlen = 128, batch_size=64):
        self.sentence_maxlen = sentence_maxlen
        self.max_vocab_size = max_vocab_size
        self.batch_size = batch_size

    def tokenize(self, dataset: Any) -> Any:
        eng, rus = dataset
        self.source_vectrorizer = TextVectorization(
            self.max_vocab_size,
            output_sequence_length=self.sentence_maxlen
        )
        self.target_vectorizer = TextVectorization(
            self.max_vocab_size,
            output_sequence_length=self.sentence_maxlen
        )
        self.source_vectrorizer.adapt(eng)
        self.target_vectorizer.adapt(rus)

        pairs = [(eng[i], rus[i]) for i in range(min(len(rus), len(eng)))]
        dataset = self.__make_pairs(pairs)
        return dataset


    def __fromat_dataset(self, eng, ru):
        eng = self.source_vectrorizer(eng)
        ru = self.target_vectorizer(ru)
        return ({
            "english": eng,
            "russian": ru[:, :-1],
        }, ru[:, 1:])
    
    def __make_pairs(self, pairs):
        eng, rus = zip(*pairs)
        eng = list(eng)
        rus = list(rus)
        dataset = tf.data.Dataset.from_tensor_slices((eng, rus))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self.__fromat_dataset, num_parallel_calls=4)
        return dataset.shuffle(2048).prefetch(16).cache()
    
    def detokenize(self, data: Any):
        vocab = self.target_vectorizer.get_vocabulary()
        vocab = dict(zip(range(len(vocab)), vocab))
        return vocab[data]
    
    def serealize(self, path: str) -> None:
        raise NotImplementedError
    
    def export_tables(self, path: str) -> None:
        raise NotImplementedError