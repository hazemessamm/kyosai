from typing import List, Tuple, Union

from jax.numpy import DeviceArray, ndarray

from kyosai.engine import model
from kyosai.engine.graph_v3 import GraphV3
from kyosai.layers.core import Input, Layer


def all_input_instances(arg):
    return all([isinstance(x, Input) for x in arg])


def all_layer_instances(arg):
    return all([isinstance(x, Layer) for x in arg])


def is_functional_params(*args, **kwargs):
    is_functional = False
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)):
            is_functional = (
                all_input_instances(arg) if i == 0 else all_layer_instances(arg)
            )
        else:
            if i == 0 and isinstance(arg, Input):
                is_functional = True
            elif isinstance(arg, Layer):
                is_functional = True

    for i, arg in enumerate(kwargs.values()):
        if isinstance(arg, (list, tuple)):
            is_functional = (
                all_input_instances(arg) if i == 0 else all_layer_instances(arg)
            )
        else:
            if i == 0 and isinstance(arg, Input):
                is_functional = True
            elif isinstance(arg, Layer):
                is_functional = True
    return is_functional


class Model(model._Model):
    def __new__(cls, *args, **kwargs):
        if is_functional_params(*args, **kwargs):
            return GraphV3(*args, **kwargs)
        elif cls == Sequential:
            return super(Model, cls).__new__(cls)
        else:
            return super(Model, cls).__new__(cls)


class Sequential(Model):
    def __init__(self, layers: Union[List[Layer], None] = None, **kwargs):
        super(Sequential, self).__init__(sequential=True, **kwargs)

        self._layers: List[Layer] = layers if layers is not None else []
        self._weights: List[Tuple] = []
        self.setup_layers()

    def setup_layers(self):
        if len(self._layers) == 0:
            return
        for i in range(1, len(self._layers)):
            self._layers[i](self._layers[i - 1])

        for layer in self._layers:
            self._weights.append(layer.weights)

    def add(self, layer: Layer):
        if isinstance(layer, Layer):
            if len(self._layers) >= 1:
                layer(self._layers[-1])
            self._layers.append(layer)
            self.weights.append(layer.weights)
        else:
            raise ValueError(
                f"add() only accepts layers subclass instances. Recieved: {layer}"
            )

    def call_with_external_weights(
        self, weights: List[Tuple], inputs: Union[ndarray, DeviceArray]
    ):
        "Accepts weights and inputs and returns predictions"
        outputs = inputs
        for weight, layer in zip(weights, self.layers):
            outputs = layer.call_with_external_weights(weight, outputs)
        return outputs

    def call(self, inputs: Union[ndarray, DeviceArray]):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
