from keras.layers import Dense, Dropout, Layer, Add, LayerNormalization
from keras.models import Sequential

class FeedForward(Layer):
    def __init__(self, dim_model, dim_hidden, dropout_rate = 0.1):
        super().__init__()
        self.dim_model = dim_model,
        self.dim_hidden = dim_hidden,
        self.dropout_rate = dropout_rate
        self.feed_forward = Sequential(
            [
                Dense(dim_hidden, activation='relu'),
                Dense(dim_model),
                Dropout(dropout_rate)
            ]
        )
        self.add = Add()
        self.norm = LayerNormalization()
    
    def call(self, x):
        return self.norm(self.add([x, self.feed_forward(x)]))
    
    def get_config(self):
        base_config = super().__init__()
        config = {
            "dim_model" : self.dim_model,
            "dim_hidden" : self.dim_hidden,
            "dropout_rate" : self.dropout_rate
        }
        return {**base_config, **config}