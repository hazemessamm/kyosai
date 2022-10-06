from typing import Tuple

from jax import numpy as jnp
from jax.numpy import DeviceArray
from kyosai.layers.base_layer import Layer


class Merge(Layer):
    """
    Merge Layer, (Layer subclass)
    Args:
        seed: random seed.
        name: name of the `Merge` layer.
    """

    def __init__(self, seed: int = None, name: str = None, **kwargs):
        super(Merge, self).__init__(seed=seed, name=name)
        self.supports_specific_axis = False
        self.supported_axis = None

    @property
    def shape(self):
        return self._output_shape

    @property
    def input_shape(self):
        return self._input_shape

    def _check_shapes(self, input_shapes):
        shapes = set(input_shapes)
        if len(shapes) != 1:
            raise ValueError(
                f"`input_shape` contains unmatched shapes. Recieved: shapes={shapes}"
            )

    def _check_axis(self, input_shapes):
        if self.supports_specific_axis:
            axis_list = set()
            for _input_shape in input_shapes:
                axis_list.add(_input_shape[self.supported_axis])

            if len(axis_list) != 1:
                raise ValueError(
                    f"`input_shape` contains unmatched axis."
                    f"Recieved: axis={self.supported_axis} and different axis are {axis_list}"
                )

    def build(self, input_shape: Tuple):
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

        self._input_shape = input_shape[0]
        self._output_shape = self.compute_output_shape(input_shape)
        self.built = True


class Concatenate(Merge):
    """
    Concatenate Layer, (Layer subclass)
    Args:
        axis: axis of the concatenation as an `int`.
    """

    def __init__(self, axis: int = -1, name: str = None, **kwargs):
        super(Concatenate, self).__init__(seed=0, name=name)
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
            sum([i[self.supported_axis] for i in input_shape]),
        )

    def build(self, input_shape: Tuple):
        super().build(input_shape)

    def concatenate_op(self, weights: Tuple, inputs: DeviceArray):
        return jnp.concatenate(inputs, axis=self.supported_axis)

    def call(self, inputs: DeviceArray, **kwargs):
        return self.concatenate_op(self.weights, inputs)

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.concatenate_op(weights, inputs)


class Add(Merge):
    """
    Add Layer, (Layer subclass)
    Args:
        name: name of the `Add` layer.
    """

    def __init__(self, name: str = None, **kwargs):
        super(Add, self).__init__(seed=0, name=name)

    @property
    def shape(self):
        return self._output_shape

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def add_op(self, weights: Tuple, inputs: DeviceArray):
        inputs = jnp.stack(inputs, axis=0)
        return jnp.sum(inputs, axis=0)

    def build(self, input_shape: Tuple):
        super().build(input_shape)

    def call(self, inputs: DeviceArray, **kwargs):
        return self.add_op(self._weights, inputs)

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.add_op(weights, inputs)
