import operator as op
from collections import OrderedDict
from typing import Dict

import jax
from kyosai.engine import generic_utils
from kyosai.engine.model import _Model
from kyosai.layers import Layer


class GraphV3(_Model):
    def __init__(self, *args, **kwargs):
        super(GraphV3, self).__init__(name=kwargs.pop("name", None))
        self.inputs, self.outputs = self._parse_args_and_kwargs(*args, **kwargs)
        self._layers_mapping: Dict[str, Layer] = OrderedDict()
        self._dependencies = OrderedDict()
        self._output_names = [output.name for output in self.outputs]
        self._create_graph()
        # generic_utils.jit_call(self)

    @property
    def output(self):
        return self.outputs[0] if len(self.outputs) == 1 else []

    def _parse_args_and_kwargs(self, *args, **kwargs):
        allowed_kwargs = {"inputs", "outputs", "input", "output"}

        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError("`inputs` and `outputs` should be passed to the model.")

        if len(args) > 2:
            raise ValueError(
                "Expected 2 args only which are `inputs/input` and `ouputs/output` of type `Layer`"
            )

        if kwargs:
            for k in kwargs.keys():
                if k not in allowed_kwargs:
                    raise ValueError(f"Unknown `{k}` found in kwargs.")

        inputs = kwargs.pop("inputs", None) or kwargs.pop("input", None)
        outputs = kwargs.pop("outputs", None) or kwargs.pop("output", None)

        if inputs is None:
            inputs = generic_utils.flatten(args[0])
        elif isinstance(inputs, (list, tuple)):
            inputs = generic_utils.flatten(inputs)
        elif isinstance(inputs, dict):
            raise TypeError("`dict` is not supported yet in inputs/input argument.")
        else:
            inputs = generic_utils.flatten(inputs)

        if outputs is None:
            outputs = generic_utils.flatten(args[1])
        elif isinstance(outputs, (list, tuple)):
            outputs = generic_utils.flatten(outputs)
        elif isinstance(outputs, dict):
            raise TypeError("`dict` is not supported yet in outputs/output argument.")
        else:
            outputs = generic_utils.flatten(outputs)

        self._output_names = [output.name for output in outputs]
        self.build_from_signature(inputs)
        return inputs, outputs

    def build_from_signature(self, inputs):
        self.input_shape = [i.shape for i in inputs]
        self.built = True

    def _create_graph(self):
        layers = jax.util.toposort(self.outputs)

        num_inputs = 0
        for layer in layers:
            if len(layer.parents) == 0:
                parents = [f"arg:{num_inputs}"]
                num_inputs += 1
            else:
                parents = [parent_layer.name for parent_layer in layer.parents]
            self._dependencies[layer.name] = parents
            self._layers_mapping[layer.name] = layer
            self._layers = layers
            self.built = True

    def call_with_external_weights(self, weights, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        if len(inputs) != len(self.inputs):
            raise Exception(
                f"Expected {len(self.inputs)} inside a list or a tuple. Recieved: {len(inputs)} input/s."
            )

        outputs = {f"arg:{i}": inputs[i] for i in range(len(inputs))}

        for weight, (layer, parent_layers) in zip(weights, self._dependencies.items()):
            incoming_inputs = op.itemgetter(*parent_layers)(outputs)
            if (
                isinstance(incoming_inputs, tuple)
                and self._layers_mapping[layer]._call_util._requires_unpacking
            ):
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    weight, *incoming_inputs, **kwargs
                )
            else:
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    weight, incoming_inputs, **kwargs
                )
        return op.itemgetter(*self._output_names)(outputs)

    def call(self, inputs, *args, **kwargs):
        return self.call_with_external_weights(self.weights, inputs, **kwargs)
