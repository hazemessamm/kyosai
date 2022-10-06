import jax  # type: ignore
from jax import lax
from jax import numpy as jnp  # type: ignore
from kyosai.layers.base_layer import Layer


class Embedding(Layer):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        embeddings_initializer="uniform",
        mask_zero=False,
        input_length=None,
        seed=None,
        trainable=True,
        dtype="float32",
        name=None,
    ):
        super(Embedding, self).__init__(
            seed=seed, trainable=trainable, dtype=dtype, name=name
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_initializer = self.get_initializer(embeddings_initializer)
        self.mask_zero = mask_zero
        self.input_length = input_length

    @property
    def shape(self):
        return self._output_shape

    def build(self, input_shape):
        self._input_shape = input_shape
        self.input_length = input_shape[-1]
        shape = (self.vocab_size, self.embedding_dim)
        self._output_shape = (self.input_length, self.embedding_dim)
        self.add_weight(
            self.seed,
            shape,
            self.embedding_initializer,
            self.dtype,
            self.name,
            self.trainable,
        )

        if self.mask_zero:
            self._weights[0][0] = 0.0

        self.built = True

    def embedding_lookup(self, weights, inputs):
        return weights[0][(inputs,)]

    def embedding_op(self, weights, inputs, training=True):
        inputs = jax.nn.one_hot(inputs, self.vocab_size, dtype=jnp.int64)
        return jnp.dot(inputs, weights[0])

    def call_with_external_weights(self, weights, inputs, training=True):
        if inputs.dtype != jnp.int64:
            if inputs.dtype in {jnp.int32, jnp.int16, jnp.int8}:
                inputs = inputs.astype("int64")
        return lax.cond(
            training,
            lambda: self.embedding_op(weights, inputs),
            lambda: self.embedding_lookup(weights, inputs),
        )

    def call(self, inputs, training=True):
        return self.call_with_external_weights(self.weights, inputs, training=training)
