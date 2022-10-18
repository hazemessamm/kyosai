from typing import Tuple

from jax import numpy as jnp
from jax.numpy import DeviceArray
from kyosai.layers.merge.base_merge import Merge


class Add(Merge):
    """
    Add Layer, (Layer subclass)
    Args:
        name: name of the `Add` layer.
    """

    def __init__(self, name: str = None, **kwargs):
        super(Add, self).__init__(seed=0, name=name)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def add_op(self, weights: Tuple, inputs: DeviceArray):
        inputs = jnp.stack(inputs, axis=0)
        return jnp.sum(inputs, axis=0)

    def build(self, input_shape: Tuple):
        super().build(input_shape)

    def call(self, inputs: DeviceArray, **kwargs):
        return self.add_op(self.weights, inputs)

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.add_op(weights, inputs)
