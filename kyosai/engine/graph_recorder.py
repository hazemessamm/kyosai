from kyosai.layers.core import Input
from kyosai.layers import base_layer


class GraphRecorder:
    """
    Created and records the operations that happens inside the
    `call` function if the user decides to create a `Model` subclass.
    """

    def __init__(self):
        self.latest_shape = None
        self.from_layer = None
        self.input_layers = []
        self.output_layers = set()

    @property
    def shape(self):
        return self.latest_shape

    @shape.setter
    def shape(self, val):
        self.latest_shape = val

    @property
    def latest_layer(self):
        return self.from_layer

    @latest_layer.setter
    def latest_layer(self, layer: base_layer.Layer):
        if layer is None:
            return
        if isinstance(layer, Input):
            self.input_layers.append(layer)

        for p in layer.parents:
            if p in self.output_layers:
                self.output_layers.remove(p)

        self.output_layers.add(layer)
        self.from_layer = layer
