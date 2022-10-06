import inspect
import operator as op
from collections import OrderedDict

import jax
from kyosai.engine import generic_utils
from kyosai.engine.model import _Model


class GraphV2(_Model):
    allowed_kwargs = {"inputs", "outputs"}

    def __init__(self, *args, **kwargs):
        super(GraphV2, self).__init__(name=kwargs.pop("name", None))
        self.inputs, self.outputs = self._check_args_and_kwargs(*args, **kwargs)
        branches, self.multiple_branches = self.get_branches()
        (
            self._dependencies,
            self._layers,
            self._params,
            self._layer_functions,
        ) = self.create_graph(branches=branches)

    def flatten(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return generic_utils.flatten(inputs)
        else:
            return [inputs]

    def _check_args_and_kwargs(self, *args, **kwargs):
        for k in kwargs.keys():
            if k not in GraphV2.allowed_kwargs:
                raise ValueError(f"unknown argument {k}")

        consumed_args = 0
        if kwargs.get("inputs", False):
            inputs = jax.tree_flatten(kwargs["inputs"])[0]
        else:
            if len(args) == 0:
                raise ValueError("Input is not passed correctly")
            else:
                inputs = jax.tree_flatten(args[consumed_args])[0]
                consumed_args += 1

        if kwargs.get("outputs", False):
            outputs = jax.tree_flatten(kwargs["outputs"])[0]
        else:
            if len(args) == 0:
                raise ValueError("Output is not passed correctly")
            else:
                outputs = jax.tree_flatten(args[consumed_args])[0]

        self._output_names = [o.name for o in outputs]
        return inputs, outputs

    def get_branches(self):
        """Returns the network branches
        Example:

        # Branch 1
        input1 = kyosai.layers.Input(...)
        conv1 = kyosai.layers.Conv2D(...)(input1)
        flatten1 = kyosai.layers.Flatten(...)(conv1)

        # Branch 2
        input2 = kyosai.layers.Input(...)
        conv1 = kyosai.layers.Conv2D(...)(input2)
        flatten2 = kyosai.layers.Flatten(...)(conv1)

        # Branch 3
        input3 = kyosai.layers.Input(...)
        conv1 = kyosai.layers.Conv2D(...)(input3)
        flatten3 = kyosai.layers.Flatten(...)(conv1)

        # Merge
        concatenate = kyosai.layers.Concatenate()([flatten1, flatten2, flatten3])

        So this method will return a list that contains 3 OrderedDict instances, each one contains a branch
        """
        branches = [OrderedDict([(_input.name, _input)]) for _input in self.inputs]
        multiple_branches = len(branches) > 1

        def _traverse(root, branch_index):
            nonlocal branches
            for layer in root._node_container.outbound_nodes:
                if layer.name not in branches[branch_index]:
                    branches[branch_index][layer.name] = layer
                    _traverse(layer, branch_index)

        for branch_index, _input in enumerate(self.inputs, start=0):
            _traverse(root=_input, branch_index=branch_index)
        # Append the main branch of the network that all the branches are added to or concatenated to etc...
        return branches, multiple_branches

    def _make_layer_function(
        self,
        layer,
        multiple_dependencies,
        is_input_layer=False,
        input_layer_index=None,
        multiple_branches=False,
    ):
        """
        Creates a function for every layer to handle it is input properly
        Args:
            - layer: Accepts `Layer` subclass.
            - multiple_dependencies: Accepts boolean value, checks whether this layer takes
                it's input from multiple other layers or not.
            - is_input_layer: If True, it will treat that layer as input and it will create a function
                that expects a tensor or a list of tensors.
            - multiple_branches: if `is_input_layer` is True, and there are multiple input layers
                then `multiple_branches` should be true to handle the input tensors properly
        """

        # If the layer expects multiple inputs other than the `inputs` argument then the inputs should be unpacked.
        requires_unpacking = (
            len(inspect.signature(layer.call_with_external_weights).parameters) > 2
        )

        if not is_input_layer:
            if multiple_dependencies:
                if requires_unpacking:
                    # returning a function that recieves multiple outputs from other layers and unpack them
                    def _call_mult_outs(params, inputs):
                        try:
                            return layer.call_with_external_weights(params, *inputs)
                        except TypeError as e:
                            raise ValueError(
                                f"Error in layer: {layer.name}, Error: {e}"
                            )

                    return _call_mult_outs
                else:
                    # returning a function that recieves multiple outputs from other layers
                    def _call_mult_deps(params, inputs):
                        try:
                            return layer.call_with_external_weights(params, inputs)
                        except TypeError as e:
                            raise ValueError(
                                f"Error in layer: {layer.name}, Error: {e}"
                            )

                    return _call_mult_deps
            else:
                # returning a function that recieves single output from other layer
                def _call_single_dep(params, inputs):
                    try:
                        return layer.call_with_external_weights(params, inputs[0])
                    except TypeError as e:
                        raise ValueError(f"Error in layer: {layer.name}, Error: {e}")

                return _call_single_dep
        else:
            if multiple_branches:
                # returning a function that recieves an input but in a list
                # and has multiple input layers
                def _call_mult_inputs(params, inputs):
                    return inputs[input_layer_index]

                return _call_mult_inputs
            else:
                # returning a function that recieves a single input for single
                # input layer
                def _call_single_input(params, inputs):
                    return inputs

                return _call_single_input

    def _loop_over_layers_in_branches(self, branches):
        for branch in branches:
            for layer_name, layer in branch.items():
                yield layer_name, layer

    def create_graph(self, branches):
        """Returns the depedencies map, layers map, parameters, layer functions

        This method takes the created branches from `get_branches` method and creates the following:
            1. Dependecies map: a dictionary that describes each layer dependencies
                (which layer depends on which layer).
            2. layers: a dictionary that has layer names as keys and layer instances as values.
            3. parameters: a list of tuples that contains the network parameters.
            4. layer functions: a dictionary that has layer names as keys and layer functions as values.
        """
        dependencies = OrderedDict()
        layers = OrderedDict()
        layer_functions = OrderedDict()
        parameters = []
        on_hold_dependencies = {}
        num_input_layers = 0
        for layer_name, layer in self._loop_over_layers_in_branches(branches):
            layers[layer_name] = layer

            if layer_name not in on_hold_dependencies:
                required_dependencies = [
                    node.name for node in layer._node_container.inbound_nodes
                ]
            else:
                required_dependencies = on_hold_dependencies[layer_name]

            if layer_name in dependencies:
                on_hold_dependencies[layer_name] = dependencies[layer_name]
                del dependencies[layer_name]
            else:
                dependencies[layer_name] = required_dependencies

            has_multiple_dependecies = len(required_dependencies) > 1
            is_input_layer = len(required_dependencies) == 0

            if is_input_layer:
                layer_fn = self._make_layer_function(
                    layer,
                    has_multiple_dependecies,
                    is_input_layer,
                    num_input_layers,
                    self.multiple_branches,
                )
                num_input_layers += 1
            else:
                layer_fn = self._make_layer_function(layer, has_multiple_dependecies)

            if not layer_fn:
                raise ValueError(
                    f"Cannot find an appropriate function for that layer {layer_name}"
                )

            layer_functions[layer_name] = layer_fn

        parameters = [layers[dep_name].params for dep_name in dependencies.keys()]
        return dependencies, layers, parameters, layer_functions

    def call_with_external_weights(self, params, inputs):
        saved_outputs = {}
        for param, (layer_name, layer_deps) in zip(params, self._dependencies.items()):

            if not layer_deps:
                saved_outputs[layer_name] = self._layer_functions[layer_name](
                    param, inputs
                )
            else:
                saved_outputs[layer_name] = self._layer_functions[layer_name](
                    param, [op.itemgetter(*layer_deps)(saved_outputs)]
                )

        if len(self.outputs) > 1:
            return [op.itemgetter(*self._output_names)(saved_outputs)]
        else:
            return saved_outputs[self._output_names[0]]

    def __call__(self, inputs, *args, **kwargs):
        return self.call_with_external_weights(self._params, inputs)
