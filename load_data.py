from typing import List
from tqdm import tqdm

class DataLoader():
    def __init__(self):
        self.pos_match = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}
        self.chunk_match = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}
        self.ne_match = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
        self.column_match = {1: self.pos_match, 2: self.chunk_match, 3: self.ne_match}

    def __open(self, path: str) -> str:
        with open(path, 'r') as file:
            content = file.read()
        return content
    
    def __transform(self, file: str) -> List:
        sentences = file.split('\n\n')
        dataset = []
        entities = []
        words = []
        for sentence in sentences:
            transformed_sentence = []
            transformed_entities = []
            entries = sentence.split('\n')
            for entry in entries:
                tags = entry.split(' ')
                transformed_sentence.append(tags[0])
                words.append(tags[0])
                transformed_entities.append(self.ne_match[tags[-1]])
            entities.append(transformed_entities)
            dataset.append(transformed_sentence)
        return dataset, entities, words
    
    def load_dataset(self, path: str) -> List:
        return self.__transform(self.__open(path))
        
    def load_embedding(self, path: str):
        embedding = open(path, 'r').read()
        words = embedding.split('\n')

        transformed_embedding = {}
        print("Loading embedding...")
        for word in tqdm(words):
            word_data = word.split(' ')
            transformed_embedding[word_data[0]] = [float(num) for num in word_data[1:]]
        print("Finished.")
        return transformed_embedding