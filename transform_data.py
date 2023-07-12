from typing import Dict, List
import numpy as np
from tqdm import tqdm

class DataTransformer():
    def __init__(self):
        self.vocab_size = 200000
        self.num_tokens = 50
    
    def transform_embedding(self, embedding: Dict, vocab: Dict):
        transformed_embedding = np.zeros((self.vocab_size, self.num_tokens))
        count = 0
        print("Trasforming embedding...")
        for i, word in tqdm(enumerate(vocab.keys())):
            if embedding.get(str(word).lower()):
                line = embedding.get(str(word).lower())
                for j, value in enumerate(line):
                    transformed_embedding[i][j] = value
                count += 1
        print(f"Finished. {count} out of {self.vocab_size} hits.")
        return transformed_embedding
    
    def transform_words(self, dataset: List[List], vocab: Dict):
        sentence_maxlen = 0
        for entry in dataset:
            if len(entry) > sentence_maxlen:
                sentence_maxlen = len(entry)
        trasformed_dataset = np.zeros((len(dataset), sentence_maxlen))
        print("Trasforming dataset...")
        for i, entry in tqdm(enumerate(dataset)):
            for j, word in enumerate(entry):
                trasformed_dataset[i][j] = vocab.get(word, 0)
        print(f"Finished. Lenght is {len(dataset)}")
        return trasformed_dataset

    def get_vocab(self, words: List):
        unique_words = list(set(words))
        words_count = {}
        print("Counting words...")
        for word in tqdm(unique_words):
            count = words.count(word)
            words_count[count] = word
        print("Finished")
        
        popular_words = list(words_count.keys())
        popular_words.sort()
        popular_words = popular_words[:self.vocab_size]

        vocab = {word: index for word, index in zip(popular_words, range(1, self.vocab_size+1))}
        return vocab