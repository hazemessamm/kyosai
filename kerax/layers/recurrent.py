from jax import numpy as jnp  # type: ignore
from kerax.layers.base_layer import Layer


class GRU(Layer):
    def __init__(
        self,
        units,
        activation="tanh",
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=False,
    ):
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_activation
        self.bias_initializer = bias_initializer
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.return_sequences = return_sequences


class LSTM(Layer):
    def __init__(self):
        pass
