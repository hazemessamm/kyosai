from typing import Tuple, Union

from jax import lax
from jax import numpy as jnp
from jax.numpy import DeviceArray

from .core import Layer


class Pooling(Layer):
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
        super(Pooling, self).__init__(
            seed=seed, trainable=False, dtype=dtype, name=name, **kwargs
        )
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self._validate_init()

        input_shape = kwargs.get("input_shape", False)
        if input_shape:
            self.build(input_shape)

    def _validate_init(self):
        if isinstance(self.pool_size, int):
            self.pool_size = (self.pool_size, self.pool_size)
        elif isinstance(self.pool_size, tuple) and len(self.pool_size) == 1:
            self.pool_size += self.pool_size

        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides += self.strides

        self.padding = self.padding.upper()

        self._pool_size = self.pool_size
        self._strides = self.strides

        self.pool_size = (1, *self.pool_size, 1)
        self.strides = (1, *self.strides, 1)

    def compute_output_shape(self):
        # lax.reduce_window_shape_tuple() does not accept batch size with None
        # so it's replaced with '1' only in this function
        input_shape = (1, *self._input_shape[1:])
        padding_vals = lax.padtype_to_pads(
            input_shape, self.pool_size, self.strides, self.padding
        )

        out_shape = lax.reduce_window_shape_tuple(
            operand_shape=input_shape,
            window_dimensions=self.pool_size,
            window_strides=self.strides,
            padding=padding_vals,
            base_dilation=(1, 1, 1, 1),
            window_dilation=(1, 1, 1, 1),
        )
        return out_shape

    @property
    def shape(self):
        return self._shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self._shape = (None, *self.compute_output_shape()[1:])
        self.built = True


class MaxPooling2D(Pooling):
    """
    MaxPool Layer, (Layer subclass)
    Params:
        - pool_size: takes the pooling size, default (2,2), accepts int or tuple
        - strides: stores size of the strides, default (1,1), accepts int or tuple
        - padding: padding for the input, accepts "valid" or "same"
        - spec: store the layer specs
        - key: stores Pseudo Random Generator Key, default PRNGKey(1)
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
            seed=seed,
            dtype=dtype,
            name=name,
            **kwargs,
        )

    def maxpool_op(self, params: Tuple, inputs: DeviceArray):
        return lax.reduce_window(
            inputs, -jnp.inf, lax.max, self.pool_size, self.strides, self.padding
        )

    def call(self, inputs: DeviceArray):
        self.output = self.maxpool_op(self.params, inputs)
        return self.output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        self.output = self.maxpool_op(params, inputs)
        return self.output


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
            **kwargs,
        )

    def avgpool_op(self, params: Tuple, inputs: DeviceArray):
        out = lax.reduce_window(
            inputs, 0.0, lax.add, self.pool_size, self.strides, self.padding
        )
        ones = jnp.ones((1, inputs.shape[1], inputs.shape[2], 1), dtype=inputs.dtype)
        window_sizes = lax.reduce_window(
            ones, 0.0, lax.add, self.pool_size, self.strides, self.padding
        )
        return lax.div(out, window_sizes)

    def call(self, inputs):
        return self.avgpool_op(self.params, inputs)

    def call_with_external_weights(self, params, inputs):
        return self.avgpool_op(params, inputs)
