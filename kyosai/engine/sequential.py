from kyosai.models import Model
from typing import Union, List, Tuple
from kyosai.layers.base_layer import Layer
from jax.numpy import ndarray, DeviceArray


class Sequential(Model):
    def __init__(self, layers: Union[List[Layer], None] = None, **kwargs):
        super(Sequential, self).__init__(sequential=True, **kwargs)

        self._layers: List[Layer] = layers if layers is not None else []
        self.setup_layers()

    def build_from_signature(self, first_layer):
        self.input_shape = first_layer.input_shape
        if len(self.input_shape) == 1:
            self.input_shape = self.input_shape[0]
        self.built = True

    def setup_layers(self):
        layers = self.layers
        for parent_layer, child_layer in zip(layers, layers[1:]):
            child_layer(parent_layer)
        self.build_from_signature(self._layers[0])

    def add(self, layer: Layer):
        if isinstance(layer, Layer):
            if len(self._layers) > 1:
                layer(self._layers[-1])
            else:
                self.build_from_signature(layer)
            self._layers.append(layer)
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
