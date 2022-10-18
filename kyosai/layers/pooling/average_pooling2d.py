from typing import Tuple

from jax import lax
from jax import numpy as jnp
from jax.numpy import DeviceArray

from kyosai.layers.pooling.base_pooling import Pooling


class AveragePooling2D(Pooling):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        seed=None,
        dtype="float32",
        name=None,
        **kwargs,
    ):
        super(AveragePooling2D, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            seed=seed,
            dtype=dtype,
            name=name,
            expand_dims=False,
        )

    def avgpool_op(self, weights: Tuple, inputs: DeviceArray):
        out = lax.reduce_window(
            inputs, 0.0, lax.add, self.pool_size, self.strides, self.padding
        )
        ones = jnp.ones((1, inputs.shape[1], inputs.shape[2], 1), dtype=inputs.dtype)
        window_sizes = lax.reduce_window(
            ones, 0.0, lax.add, self.pool_size, self.strides, self.padding
        )
        return lax.div(out, window_sizes)

    def call(self, inputs, **kwargs):
        return self.avgpool_op(self.weights, inputs)

    def call_with_external_weights(self, weights, inputs, **kwargs):
        return self.avgpool_op(weights, inputs)
