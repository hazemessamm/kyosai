from typing import Tuple, Union, List
from jax.numpy import DeviceArray
from kyosai.layers.base_layer import Layer


class LayerNormalization(Layer):
    def __init__(
        self,
        seed: int = None,
        trainable: bool = True,
        axis: int = -1,
        name: str = None,
        **kwargs,
    ):
        super(LayerNormalization, self).__init__(
            seed=seed, trainable=trainable, **kwargs
        )
        self.axis = axis
        self.name = name

    # TODO
    def build(self, input_shape):
        pass

    # TODO
    def call(self, inputs: DeviceArray, weights: Tuple = None, **kwargs):
        pass

    # TODO
    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        pass
