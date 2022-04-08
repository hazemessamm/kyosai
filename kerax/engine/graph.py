from collections import OrderedDict

import jax
from jax import lax
from jax import numpy as jnp

from . import generic_utils


class GraphV2:
    allowed_kwargs = {'inputs', 'outputs'}
    def __init__(self, *args, **kwargs):
        self.inputs, self.outputs = self._check_args_and_kwargs(*args, **kwargs)
        branches, self.multiple_branches = self.get_branches()
        self.dependencies, self.layers, self.params, self.layer_functions = self.create_graph(branches)

    def flatten(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return generic_utils.flatten(inputs)
        else:
            return [inputs]


    def _check_args_and_kwargs(self, *args, **kwargs):
        for k in kwargs.keys():
            if k not in GraphV2.allowed_kwargs:
                raise Exception(f'unknown argument {k}')

        
        consumed_args = 0
        if kwargs.get('inputs', False):
            inputs = jax.tree_flatten(kwargs['inputs'])[0]
        else:
            if len(args) == 0:
                raise Exception('Input is not passed correctly')
            else:
                inputs = jax.tree_flatten(args[consumed_args])[0]
                consumed_args += 1

        if kwargs.get('outputs', False):
            outputs = jax.tree_flatten(kwargs['outputs'])[0]
        else:
            if len(args) == 0:
                raise Exception('Output is not passed correctly')
            else:
                outputs = jax.tree_flatten(args[consumed_args])[0]
        return inputs, outputs



    def get_branches(self):
        branches = [OrderedDict([(_input.name, _input)]) for _input in self.inputs]
        multiple_branches = len(branches) > 1

        main_layers = OrderedDict()
        def _traverse(root, branch_index):
            nonlocal branches
            for layer in root._node_container.outbound_nodes:
                if layer.name not in branches[branch_index]:
                    if branch_index > 0 and layer.name in branches[branch_index-1] and layer.name not in main_layers:
                        del(branches[branch_index-1][layer.name])
                        main_layers[layer.name] = layer
                    elif layer.name in main_layers:
                        continue
                    else:
                        branches[branch_index][layer.name] = layer
                    _traverse(layer, branch_index)

        for branch_index, _input in enumerate(self.inputs, start=0):
            _traverse(root=_input, branch_index=branch_index)

        # Append the main branch of the network that all the branches are added to or concatenated to etc...
        branches.append(main_layers)
        return branches, multiple_branches


    def _make_function(self, layer, multiple_dependencies, is_input_layer=False, input_layer_index=None, multiple_branches=False):
        requires_unpacking = layer.call_with_external_weights.__code__.co_argcount > 3
        _call = None

        if not is_input_layer:
            if requires_unpacking and multiple_dependencies:
                # returning a function that recieves multiple outputs from other layers and unpack them
                def _call_mult_outs(params, inputs):
                    return layer.call_with_external_weights(params, *inputs)
                return _call_mult_outs
            elif not requires_unpacking and multiple_dependencies:
                # returning a function that recieves multiple outputs from other layers
                def _call_mult_deps(params, inputs):
                    return layer.call_with_external_weights(params, inputs)
                return _call_mult_deps
            elif not multiple_dependencies:
                # returning a function that recieves single output from other layer
                def _call_single_dep(params, inputs):
                    return layer.call_with_external_weights(params, inputs[0])
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
        return _call

    def create_graph(self, branches):
        dependencies = OrderedDict()
        layers = OrderedDict()
        parameters = []
        layer_functions = OrderedDict()
        num_input_layers = 0
        for branch in branches:
            for layer_name, layer in branch.items():
                layers[layer_name] = layer
                required_outputs_from = [node.name for node in layer._node_container.inbound_nodes]
                dependencies[layer_name] = required_outputs_from
                if len(required_outputs_from) == 0:
                    layer_fn = self._make_function(layer, len(required_outputs_from) > 1, True, num_input_layers, self.multiple_branches)
                    num_input_layers += 1
                else:
                    layer_fn = self._make_function(layer, len(required_outputs_from) > 1)

                if layer_fn is not None:
                    layer_functions[layer_name] = layer_fn
                else:
                    raise Exception(f'Cannot find an appropriate function for that layer {layer_name}')
                parameters.append(layer.params)
        return dependencies, layers, parameters, layer_functions

    
    def update_params(self, new_params):
        for param, layer_name in zip(new_params, self.dependencies.keys()):
            self.layers[layer_name].update_weights(param)
    
    def call_with_external_weights(self, params, inputs):
        saved_outputs = {}
        for param, (key, val) in zip(params, self.dependencies.items()):

            if len(val) == 0:
                saved_outputs[key] = self.layer_functions[key](param, inputs)
            else:
                saved_outputs[key] = self.layer_functions[key](param, [saved_outputs[v] for v in val])
        
        if len(self.outputs) > 1:
            return jnp.stack([saved_outputs[layer.name] for layer in self.outputs], axis=1)
        else:
            return saved_outputs[self.outputs[0].name]

    def __call__(self, inputs):
        return self.call_with_external_weights(self.params, inputs)
