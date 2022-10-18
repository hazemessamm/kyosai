from jax import nn
from jax import numpy as jnp

from kyosai.layers.core import Dense
from kyosai.layers.base_layer import Layer


class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def attention_op(self, query, key, value, mask=None):
        d_model = query.shape[-1]
        scores = jnp.divide(
            jnp.matmul(query, key.transpose(0, 2, 1)), jnp.sqrt(d_model)
        )
        if mask is not None:
            scores = jnp.matmul(scores, mask)
        attention = nn.softmax(scores, axis=-1)
        attention = jnp.matmul(attention, value)
        return attention

    def build(self, input_shape):
        self._shape, self.input_shape = input_shape, input_shape

    def call(self, query, key, value, mask=None):
        return self.attention_op(query, key, value, mask)


class MultiHeadAttention(Layer):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        activation="relu",
        use_bias=True,
        seed=None,
        trainable=True,
        dtype="float32",
        name=None,
    ):

        super(MultiHeadAttention, self).__init__(
            seed=seed, trainable=trainable, dtype=dtype, name=name
        )

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim is not divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.use_bias = use_bias
        self.activation = activation

        self.q_dense = Dense(
            embedding_dim,
            activation=activation,
            use_bias=use_bias,
            trainable=trainable,
            seed=seed,
        )
        self.k_dense = Dense(
            embedding_dim,
            activation=activation,
            use_bias=use_bias,
            trainable=trainable,
            seed=seed,
        )
        self.v_dense = Dense(
            embedding_dim,
            activation=activation,
            use_bias=use_bias,
            trainable=trainable,
            seed=seed,
        )
        self.o_dense = Dense(
            embedding_dim,
            activation=activation,
            use_bias=use_bias,
            trainable=trainable,
            seed=seed,
        )

    # TODO
    def compute_mask(self, x):
        pass

    def attention_op(self, query, key, value, mask=None):
        d_model = query.shape[-1]
        scores = jnp.divide(
            jnp.matmul(query, key.transpose(0, 2, 1)), jnp.sqrt(d_model)
        )
        if mask is not None:
            scores = jnp.matmul(scores, mask)
        attention = nn.softmax(scores, axis=-1)
        attention = jnp.matmul(attention, value)
        return attention

    def build(self, query_shape, value_shape, key_shape=None):
        query_dim, value_dim = query_shape, value_shape

        if key_shape is None:
            key_dim = value_dim
        else:
            key_dim = key_shape

        self.q_dense.build(query_dim)
        self.k_dense.build(key_dim)
        self.v_dense.build(value_dim)
        self.o_dense.build(self.q_dense.shape)
        self._input_shape = query_dim
        self.built = True
        self._shape = self.o_dense.shape

    @property
    def shape(self):
        return self._shape

    def _transpose_qkv(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.num_heads, -1)
        inputs = inputs.transpose(0, 2, 1, 3)
        return inputs.reshape(-1, inputs.shape[2], inputs.shape[3])

    def _transpose_output(self, inputs):
        inputs = inputs.reshape(-1, self.num_heads, inputs.shape[1], inputs.shape[2])
        inputs = inputs.transpose(0, 2, 1, 3)
        return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)

    def call_with_external_weights(self, weights, query, key, value, mask=None):
        queries = self._transpose_qkv(
            self.q_dense.call_with_external_weights(weights[0], query)
        )
        keys = self._transpose_qkv(
            self.k_dense.call_with_external_weights(weights[1], key)
        )
        values = self._transpose_qkv(
            self.v_dense.call_with_external_weights(weights[2], value)
        )
        attention = self.attention_op(queries, keys, values, mask)
        attention = self._transpose_output(attention)
        attention = self.o_dense.call_with_external_weights(weights[3], attention)
        return attention

    def call(self, query, key, value, mask=None):
        return self.call_with_external_weights(self.weights, query, key, value, mask)
