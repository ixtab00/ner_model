from keras.layers import Layer, MultiHeadAttention, LayerNormalization, Add

class BaseAttention(Layer):
    def __init__(self, num_heads, dim_model, dropout_rate = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dropout_rate = dropout_rate
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim_model, 
            dropout=dropout_rate
        )
        self.norm = LayerNormalization(epsilon=1e-6)
        self.add = Add()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
            "dim_model": self.dim_model
        }
        return {**base_config, **config}

class CrossAttention(BaseAttention):
    def call(self, x, context):
        att_output = self.attention(
            query = x,
            key = context,
            value = context
        )
        x = self.add([att_output, x])
        x = self.norm(x)
        return x
    
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        att_output = self.attention(
            query = x,
            key = x,
            value = x,
            use_causal_mask = True
        )
        x = self.add([att_output, x])
        x = self.norm(x)
        return x
    
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        att_output = self.attention(
            query = x,
            key = x,
            value = x
        )
        x = self.add([att_output, x])
        x = self.norm(x)
        return x