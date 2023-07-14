from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences
from json import loads

class PipelineStage(ABC):
    def __init__(self):
        self.pass_to = []
        self.temp_data = []

    @abstractmethod
    def compute(self, data):
        return data

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
                data = stage.compute(data)
            
        return data
        

class PSEncode(PipelineStage):
    def __init__(self, words_encoding: Dict, decoder_stage: PipelineStage):
        super().__init__()
        self.words_encoding = words_encoding
        self.unk_index = words_encoding.get("UNK")

        self.link(decoder_stage)
        #with open(path_to_model_config) as file:
            #self.model_config = loads(file.read())
        self.model_config = {'maxlen':35}

    def compute(self, data: str):
        assert type(data) == type('')
        data = data.split(' ')
        while '' in data:
            data.remove('')

        self.pass_data(data)
        data = [[self.words_encoding.get(word, self.unk_index) for word in data],]
        data = pad_sequences(data, maxlen=self.model_config['maxlen'], padding='post', value=self.words_encoding["PAD"])[0]
        return data.reshape((1, self.model_config['maxlen']))
    

class PSCompute(PipelineStage):
    def __init__(self, path_to_weights: str):
        super().__init__()
        self.model = keras.models.load_model(path_to_weights)

    def compute(self, data: np.ndarray):
        return self.model.predict(data)
    

class PSDecode(PipelineStage):
    def __init__(self, tags_encoding: Dict):
        super().__init__()
        self.index_to_tag = {index: tag for tag, index in tags_encoding.items()}
    
    def compute(self, data):
        preds = np.argmax(data, axis=-1)
        preds = np.reshape(preds, (35))
        decoded_preds = []
        for pred in preds:
            decoded_preds.append(self.index_to_tag[pred])
        output = [(self.temp_data[0][i], decoded_preds[i]) for i in range(len(self.temp_data[0]))]
        self.temp_data.clear()
        return output



pipe = Pipeline()