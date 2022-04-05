from jax import nn
from jax import numpy as jnp

from .core import Dense, Layer


class Attention(Layer):
    def attention_op(self, query, key, value, mask=None):
        d_model = query.shape[-1]
        scores = jnp.divide(jnp.matmul(query, key.transpose(0, 2, 1)), jnp.sqrt(d_model))
        if mask is not None:
            scores = jnp.matmul(scores, mask)
        attention = nn.softmax(scores, axis=-1)
        attention = jnp.matmul(attention, value)
        return attention

    def build(self, input_shape):
        self._shape, self.input_shape = input_shape, input_shape

    def call(self, query, key, value, mask=None):
        return self.attention_op(query, key, value, mask)


class MultiHeadAttention(Attention):
    def __init__(self, embedding_dim, num_heads, activation='relu', use_bias=True, key=None, trainable=True, dtype='float32', name=None):
        if embedding_dim % num_heads != 0:
            raise Exception('embedding_dim is not divisible by num_heads')
            
        super(MultiHeadAttention, self).__init__(key=key, trainable=trainable, dtype=dtype, name=name)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.activation = activation

        
        
        self.q = Dense(embedding_dim, activation=activation, use_bias=use_bias, trainable=trainable)
        self.k = Dense(embedding_dim, activation=activation, use_bias=use_bias, trainable=trainable)
        self.v = Dense(embedding_dim, activation=activation, use_bias=use_bias, trainable=trainable)
        self.o = Dense(embedding_dim, activation=activation, use_bias=use_bias, trainable=trainable)

    # TODO
    def compute_mask(self, x):
        pass

    def build(self, input_shape):
        query_dim, key_dim, value_dim = input_shape
        self.q.build(query_dim)
        self.k.build(key_dim)
        self.v.build(value_dim)
        self.o.build(query_dim)
        self._input_shape = query_dim
    
    @property
    def shape(self):
        return self._input_shape

    def _transpose_qkv(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_heads, -1)  
        inputs = inputs.transpose(0, 2, 1, 3)
        return inputs.reshape(-1, inputs.shape[2], inputs.shape[3])

    def _transpose_output(self, inputs):
        inputs = inputs.reshape(-1, self.num_heads, inputs.shape[1], inputs.shape[2])
        inputs = inputs.transpose(0, 2, 1, 3)
        return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)

    def call(self, query, key, value, mask=None):
        
        queries = self._transpose_qkv(self.q(query))
        keys = self._transpose_qkv(self.k(key))
        values = self._transpose_qkv(self.v(value))
        attention = self.attention_op(queries, keys, values, mask)
        attention = self._transpose_output(attention)
        attention = self.o(attention)
        return attention
    
    def call_with_external_weights(self, params, query, key, value, mask=None):
        queries = self._transpose_qkv(self.q.call_with_external_weights(params[0], query))
        keys = self._transpose_qkv(self.k.call_with_external_weights(params[1], key))
        values = self._transpose_qkv(self.v.call_with_external_weights(params[2], value))
        attention = self.attention_op(queries, keys, values, mask)
        attention = self._transpose_output(attention)
        attention = self.o.call_with_external_weights(params[3], attention)
        return attention
