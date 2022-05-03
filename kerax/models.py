from typing import Any, List, Tuple, Union

from jax.numpy import DeviceArray, ndarray

from kerax.engine import model
from kerax.engine.graph import GraphV2
from kerax.layers.core import Layer


def is_functional_params(*args, **kwargs):
    is_functional = False
    for arg in args:
        if isinstance(arg, Layer):
            is_functional = True

    for val in kwargs.values():
        if isinstance(val, Layer):
            is_functional = True
    return is_functional


class Model(model._Model):
    def __new__(cls, *args, **kwargs):
        if is_functional_params(*args, **kwargs):
            return GraphV2(*args, **kwargs)
        elif cls == Sequential:
            return super(Model, cls).__new__(cls)
        else:
            return super(Model, cls).__new__(cls)


class Sequential(Model):
    def __init__(self, layers: Union[List[Layer], None] = None, **kwargs):
        super(Sequential, self).__init__(sequential=True, **kwargs)

        self._layers: List[Layer] = layers if layers is not None else []
        self._params: List[Tuple] = []
        self.setup_layers()

    def setup_layers(self):
        if len(self._layers) == 0:
            return
        for i in range(1, len(self._layers)):
            self._layers[i](self._layers[i - 1])

        for layer in self._layers:
            self._params.append(layer.params)

    def add(self, layer: Layer):
        if isinstance(layer, Layer):
            if len(self._layers) >= 1:
                layer(self._layers[-1])
            self._layers.append(layer)
            self.params.append(layer.params)
        else:
            raise ValueError(
                f"add() only accepts layers subclass instances. Recieved: {layer}"
            )

    def call_with_external_weights(
        self, params: List[Tuple], inputs: Union[ndarray, DeviceArray]
    ):
        "Accepts params and inputs and returns predictions"
        outputs = inputs
        for param, layer in zip(params, self.layers):
            outputs = layer.call_with_external_weights(param, outputs)
        return outputs

    def __call__(self, inputs: Union[ndarray, DeviceArray]):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
