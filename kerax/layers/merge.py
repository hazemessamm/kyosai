from typing import Tuple

from jax import numpy as jnp
from jax.numpy import DeviceArray

from .core import Layer


class Merge(Layer):
    def __init__(self, seed: int = None, name: str = None, **kwargs):
        super(Merge, self).__init__(seed=seed, name=name, **kwargs)
        self.supports_different_shapes = True
        self.supports_specific_axis = False
        self.supported_axis = None

    @property
    def shape(self):
        return self._output_shape

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape()

    @property
    def input_shape(self):
        return self._input_shape

    def _check_shapes(self, input_shapes):
        if not self.supports_different_shapes:
            shapes = set(input_shapes)
            if len(shapes) != 1:
                raise ValueError(
                    f"`input_shape` contains unmatched shapes. Recieved: shapes={shapes}"
                )

    def _check_axis(self, input_shapes):
        if self.supports_specific_axis:
            axis_list = {}
            for _input_shape in input_shapes:
                axis_list.add(_input_shape[self.supported_axis])

            if len(axis_list) != 1:
                raise ValueError(
                    f"`input_shape` contains unmatched axis. Recieved: axis={self.supported_axis} and different axis are {axis_list}"
                )

    def build(self, input_shape: Tuple):
        if self.supports_different_shapes and self.supports_specific_axis:
            raise ValueError(
                "Cannot set both `supports_different_shapes` and `supports_specific_axis` to True"
            )
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError(
                f"`input_shape` should be a list of input shapes. Recieved: input_shape={input_shape}"
            )
        elif len(input_shape) <= 1:
            raise ValueError(
                f"`input_shape` should be a list with length bigger than 1. Recieved: input_shape={input_shape}"
            )
        else:
            self._check_shapes(input_shape)
            self._check_axis(input_shape)

        self._input_shape = input_shape
        self._output_shape = self.compute_output_shape()
        self.built = True


class Concatenate(Merge):
    def __init__(self, axis: int = -1, name: str = None, **kwargs):
        super(Concatenate, self).__init__(seed=0, name=name, **kwargs)
        self.supports_different_shapes = False
        self.supports_specific_axis = True
        self.supported_axis = axis

    @property
    def shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def compute_output_shape(self, input_shape):
        return (
            *input_shape[0][:-1],
            sum([i[self.axis] for i in input_shape]),
        )

    def build(self, input_shape: Tuple):
        super().build(input_shape)

    def concatenate_op(self, params: Tuple, inputs: DeviceArray):
        return jnp.concatenate(inputs, axis=self.axis)

    def call(self, inputs: DeviceArray):
        return self.concatenate_op(self.params, inputs)

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        return self.concatenate_op(params, inputs)


class Add(Merge):
    def __init__(self, name: str = None, **kwargs):
        super(Add, self).__init__(seed=0, name=name, **kwargs)
        self.supports_different_shapes = False

    @property
    def shape(self):
        return self._output_shape

    @property
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def add_op(self, params: Tuple, inputs: DeviceArray):
        inputs = jnp.stack(inputs, axis=0)
        return jnp.sum(inputs, axis=0)

    def build(self, input_shape: Tuple):
        super().build()

    def call(self, inputs: DeviceArray):
        return self.add_op(self._params, inputs)

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        return self.add_op(params, inputs)
