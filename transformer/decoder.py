from transformer.attention import CausalSelfAttention, CrossAttention
from transformer.embedding import PositionalEmbedding
from transformer.feed_forward import FeedForward
from keras.layers import Layer, Dropout

class DecoderLayer(Layer):
    def __init__(
            self,
            num_heads: int,
            dim_model: int, 
            dim_hidden: int, 
            dropout_rate = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.dropout_rate = dropout_rate

        self.causal_attention = CausalSelfAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            dropout_rate=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            dim_model=dim_model,
            dropout_rate=dropout_rate
        )
        self.feed_forward = FeedForward(
            dim_model=self.dim_model,
            dim_hidden=self.dim_hidden,
            dropout_rate=self.dropout_rate
        )
    
    def call(self, x, context):
        x = self.causal_attention(x)
        x = self.cross_attention(x, context=context)
        x = self.feed_forward(x)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_heads": self.num_heads,
            "dim_model": self.dim_model,
            "dim_hidden": self.dim_hidden,
            "dropout_rate": self.dropout_rate
        }
        return {**base_config, **config}
    
class Decoder(Layer):
    def __init__(
            self,
            num_layers: int,
            num_heads: int,
            dim_model: int,
            dim_hidden: int,
            vocab_size: int,
            dropout_rate = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.embedding = PositionalEmbedding(vocab_size, dim_model)
        self.dropout = Dropout(dropout_rate)
        self.layers = [
            DecoderLayer(
                num_heads=self.num_heads,
                dim_model=self.dim_model,
                dim_hidden=self.dim_hidden,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]
    
    def call(self, x, context):
        x = self.embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, context=context)
        return x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dim_model": self.dim_model,
            "dim_hidden": self.dim_hidden,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate
        }
        return {**base_config, **config}