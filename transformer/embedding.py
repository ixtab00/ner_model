from keras.layers import Layer, Embedding
import tensorflow as tf
import numpy as np

POSITIONAL_ENCODING_ANGLE_BASE = 10000
POSITIONAL_ENCODING_LENGTH = 2048

class PositionalEmbedding(Layer):
    def __init__(self, vocab_size: int, dim_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.embedding = Embedding(vocab_size, dim_model, mask_zero=True)
        self.positional_encoding = self.__positional_encoding()

    def __positional_encoding(
            self,
            angle_base = POSITIONAL_ENCODING_ANGLE_BASE,
            length = POSITIONAL_ENCODING_LENGTH
    ):
        depth = self.dim_model/2

        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]

        angle_rates = 1/(angle_base**depths)
        angle_rads = positions * angle_rates

        encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

        return tf.cast(encoding, dtype=tf.float32)

    def compute_mask(self, *args, **kwargs):
        self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dim_model, tf.float32))
        x = x + self.positional_encoding[tf.newaxis, :length, :]
        return x
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "dim_model": self.dim_model,
            "vocab_size": self.vocab_size
        }
        return {**base_config, **config}
