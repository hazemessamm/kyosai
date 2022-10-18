from typing import Tuple

from jax import numpy as jnp
from jax.numpy import DeviceArray
from kyosai.layers.merge.base_merge import Merge


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
