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
        self._params = []
        self._output_names = [output.name for output in self.outputs]
        self._create_graph()
        generic_utils.jit_call(self)

    @property
    def output(self):
        return self.outputs[0] if len(self.outputs) == 1 else []

    def _parse_args_and_kwargs(self, *args, **kwargs):
        allowed_kwargs = {"inputs", "outputs"}

        if len(args) == 0 and len(kwargs) == 0:
            raise ValueError("`inputs` and `outputs` should be passed to the model.")

        if len(args) > 2:
            raise ValueError(
                "Expected 2 args only which are `inputs` and `ouputs` of type layers"
            )

        unknown_kwargs = [k for k in kwargs.keys() if k not in allowed_kwargs]
        if len(unknown_kwargs) > 0:
            raise ValueError(f"unknown argument {unknown_kwargs}")

        consumed_args = 0
        if kwargs.get("inputs", False):
            inputs = generic_utils.flatten(kwargs["inputs"])
        else:
            inputs = generic_utils.flatten(args[consumed_args])
            consumed_args += 1

        if kwargs.get("outputs", False):
            outputs = generic_utils.flatten(kwargs["outputs"])
        else:
            outputs = generic_utils.flatten(args[consumed_args])

        self._output_names = op.attrgetter("name")(*outputs)
        return inputs, outputs

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
            self._params.append(layer.params)
            self._layers = layers
            self.built = True

    def call_with_external_weights(self, params, inputs, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = {f"arg:{i}": inputs[i] for i in range(len(inputs))}

        for param, (layer, parent_layers) in zip(params, self._dependencies.items()):
            incoming_inputs = op.itemgetter(*parent_layers)(outputs)
            if (
                isinstance(incoming_inputs, tuple)
                and self._layers_mapping[layer]._call_util._requires_unpacking
            ):
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    param, *incoming_inputs, **kwargs
                )
            else:
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    param, incoming_inputs, **kwargs
                )
        return op.itemgetter(*self._output_names)(outputs)

    def call(self, inputs):
        pass

    def __call__(self, inputs, **kwargs):
        return self.call_with_external_weights(self.params, inputs, **kwargs)
