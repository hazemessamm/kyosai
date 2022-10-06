from typing import Tuple

from jax.example_libraries import stax
from jax.numpy import DeviceArray
from kyosai.layers.base_layer import Layer


# TODO: Should remove stax and re-write it.
class BatchNormalization(Layer):
    def __init__(
        self,
        seed: int = None,
        trainable: bool = True,
        axis: int = -1,
        name: str = None,
        **kwargs,
    ):
        super(BatchNormalization, self).__init__(
            seed=seed, trainable=trainable, **kwargs
        )
        self.axis = axis
        self.name = name

    def build(self, input_shape: tuple):
        init_fun, self.apply_fn = stax.BatchNorm(axis=self.axis)
        self.shape, self._weights = init_fun(rng=self.key, input_shape=input_shape)
        self.input_shape = self.shape
        self.built = True

    def call(self, inputs: DeviceArray, **kwargs):
        output = self.apply_fn(weights=self._weights, x=inputs)
        return output

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        output = self.apply_fn(weights=weights, inputs=inputs)
        return output
