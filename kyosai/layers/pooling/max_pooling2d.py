from typing import Tuple, Union

from jax import lax
from jax import numpy as jnp
from jax.numpy import DeviceArray

from kyosai.layers.pooling.base_pooling import Pooling


class MaxPooling2D(Pooling):
    """
    MaxPool Layer, (Layer subclass)
    Args:
        pool_size: takes the pooling size, default (2,2), accepts int or tuple
        strides: stores size of the strides, default (1,1), accepts int or tuple
        padding: padding for the input, accepts "valid" or "same"
        spec: store the layer specs
        key: stores Pseudo Random Generator Key, default PRNGKey(1)
    """

    def __init__(
        self,
        pool_size: Union[int, Tuple] = (2, 2),
        strides: Union[int, Tuple] = (2, 2),
        padding: str = "valid",
        seed: int = None,
        dtype="float32",
        name=None,
        **kwargs,
    ):
        super(MaxPooling2D, self).__init__(
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            dims=2,
            seed=seed,
            dtype=dtype,
            name=name,
        )

    def maxpool_op(self, weights: Tuple, inputs: DeviceArray):
        return lax.reduce_window(
            inputs, -jnp.inf, lax.max, self.pool_size, self.strides, self.padding
        )

    def call(self, inputs: DeviceArray, **kwargs):
        output = self.maxpool_op(self.weights, inputs)
        return output

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        output = self.maxpool_op(weights, inputs)
        return output


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
