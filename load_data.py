import pandas as pd

class DataLoader():
    def open_csv(self, path: str, encoding: str, tag_header: str, words_header: str, sentence_header: str):
        dataframe = pd.read_csv(path, encoding=encoding)
        dataframe = dataframe.fillna(method='ffill')

        unique_words = list(set(dataframe[words_header].values))
        unique_tags = list(set(dataframe[tag_header].values))

        sentences = dataframe.groupby(f"{sentence_header} #").apply(lambda data: [(word, tag) for word, tag in zip(data[words_header].values.tolist(),
                                                                                             data[tag_header].values.tolist())])
        
        return unique_words, unique_tags, sentences.tolist()