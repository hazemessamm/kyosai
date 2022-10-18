from typing import Tuple, Union, List
from jax.numpy import DeviceArray
from kyosai.layers.base_layer import Layer


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

    def build(self, input_shape):
        self.input_shape = input_shape
        self.gamma = self.add_weight(
            key=self.seed,
            shape=input_shape[-1],
            initializer="ones",
            dtype=self.dtype,
            name=self.name + "gamma",
            trainable=self.trainable,
        )
        self.beta = self.add_weight(
            key=self.seed,
            shape=input_shape[-1],
            initializer="zeros",
            dtype="float32",
            name=self.name + "beta",
            trainable=self.trainable,
        )

        self.running_mean = self.add_weight(
            key=self.seed,
            shape=input_shape[-1],
            initializer="zeros",
            dtype=self.dtype,
            name=self.name + "running_mean",
            trainable=False,
        )
        self.running_variance = self.add_weight(
            key=self.seed,
            shape=input_shape[-1],
            initializer="ones",
            dtype=self.dtype,
            name=self.name + "running_variance",
            trainable=False,
        )

    # TODO
    def call(self, inputs: DeviceArray, weights: Tuple = None, **kwargs):
        pass

    # TODO
    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        pass
