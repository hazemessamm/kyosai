from .core import Layer, Dense
from kerax import backend


class MultiHeadAttention(Layer):
    def __init__(self, embedding_dim, num_heads, activation='relu', key=None, trainable=True, dtype='float32', name=None):
        super(MultiHeadAttention, self).__init__(key=key, trainable=trainable, dtype=dtype, name=name)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        if embedding_dim % num_heads != 0:
            raise Exception('embedding_dim is not divisible by num_heads')
        
        # TODO: Continue working on the Dense layer 
        self.qkv = Dense(embedding_dim*3, activation=activation)