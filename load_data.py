import pandas as pd

class DataLoader():
    def open_csv(self, path: str):
        dataframe = pd.read_csv(path, encoding="latin1")
        dataframe = dataframe.fillna(method='ffill')

        unique_words = list(set(dataframe['Word'].values))
        unique_tags = list(set(dataframe['Tag'].values))

        sentences = dataframe.groupby("Sentence #").apply(lambda data: [(word, tag) for word, tag in zip(data['Word'].values.tolist(),
                                                                                             data['Tag'].values.tolist())])
        
        return unique_words, unique_tags, sentences.tolist()