from kyosai.models import Model
from typing import Union, List, Tuple
from kyosai.layers.base_layer import Layer
from jax.numpy import ndarray, DeviceArray


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
