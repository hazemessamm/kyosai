from collections import deque, namedtuple, OrderedDict

from jax import numpy as jnp
from kerax.layers.core import Input


class GraphV2:
    allowed_kwargs = {'input', 'inputs', 'output', 'outputs'}
    def __init__(self, *args, **kwargs):
        self.inputs, self.outputs = self._check_args_and_kwargs(*args, **kwargs)
        branches, self.multiple_branches = self.get_branches()
        self.layers, self.dependencies, self.params = self.create_graph(branches)

    def flatten(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return list(itertools.chain(inputs))
        else:
            return [inputs]
    
    def _check_args_and_kwargs(self, *args, **kwargs):
        inputs, outputs = self._check_args(*args)

        if not inputs and not outputs:
            inputs, outputs = self._check_kwargs(**kwargs)
            if not inputs and not outputs:
                raise Exception('Cannot create a Model without inputs or outputs')
            else:
                return inputs, outputs
        else:
            return inputs, outputs


    def _check_args(self, *args):
        if len(args) == 0:
            return None, None
        inputs = self.flatten(args[0])
        outputs = self.flatten(args[1])
        return inputs, outputs


    def _check_kwargs(self, **kwargs):
        if len(kwargs) == 0:
            return None, None
        
        for k, v in kwargs.items():
            if k not in GraphV2.allowed_kwargs:
                raise Exception(f'Unknown argument {k}')
            else:
                if k == 'input' or k == 'inputs':
                    if k == 'input' and isinstance(v, (list, tuple)):
                        raise Exception(f'Expected a single element found an element in a {type(v)}')
                    elif k == 'inputs' and not isinstance(v, (list, tuple)):
                        raise Exception(f'Expected the element to be found in a list or a tuple found {type(v)}')
                    else:
                        inputs = self.flatten(v)
                    
                elif k == 'output' or k == 'outputs':
                    if k == 'output' and isinstance(v, (list, tuple)):
                        raise Exception(f'Expected a single element found an element in a {type(v)}')
                    elif k == 'outputs' and not isinstance(v, (list, tuple)):
                        raise Exception(f'Expected the element to be found in a list or a tuple found {type(v)}')
                    else:
                        outputs = self.flatten(v)
        return inputs, outputs


    def get_branches(self):
        branches = [OrderedDict([(_input.name, _input)]) for _input in self.inputs]
        multiple_branches = len(branches) > 1

        main_layers = OrderedDict()
        def _traverse(root, branch_index, visited):
            nonlocal branches
            for r in root._node_container.outbound_nodes:
                if r.name not in visited:
                    visited.add(r.name)
                    if branch_index > 0 and r.name in branches[branch_index-1] and r.name not in main_layers:
                        del(branches[branch_index-1][r.name])
                        main_layers[r.name] = r
                    elif r.name in main_layers:
                        continue
                    else:
                        branches[branch_index][r.name] = r
                    
                    _traverse(r, branch_index, visited)

        for branch_index, _input in enumerate(self.inputs, start=0):
            _traverse(root=_input, branch_index=branch_index, visited=set())

        return branches, multiple_branches

    def create_graph(self, branches):
        requires_output_from = OrderedDict()
        # requires_output_to = OrderedDict()
        layers = OrderedDict([(layer_name, layer) for branch in branches for layer_name, layer in branch.items()])
        for layer_name, layer in layers.items():
            required_outputs_from = [node.name for node in layer._node_container.inbound_nodes]
            requires_output_from[layer_name] = required_outputs_from
            # required_outputs_to = [node.name for node in layer._node_container.outbound_nodes]
            # requires_output_to[layer_name] = required_outputs_to

        params = [layers[layer_name].params for layer_name in requires_output_from.keys()]
        return layers, requires_output_from, params

    
    def update_params(self, new_params):
        for param, layer_name in zip(new_params, self.dependencies.keys()):
            self.layers[layer_name].update_weights(param)
    

    def call_with_external_weights(self, params, tensors):
        saved_outputs = OrderedDict()
        consumed_indices = 0
        for param, (key, val) in zip(params, self.dependencies.items()):
            if len(val) == 0:
                saved_outputs[key] = self.layers[key].call_with_external_weights(param, tensors[consumed_indices])
                consumed_indices += 1
            else:
                if len(val) == 1:
                    saved_outputs[key] = self.layers[key].call_with_external_weights(param, saved_outputs[val[0]])
                else:
                    requires_unpacking = self.layers[key].call_with_external_weights.__code__.co_argcount > 3
                    if requires_unpacking:
                        saved_outputs[key] = self.layers[key].call_with_external_weights(param, *[saved_outputs[v] for v in val])
                    else:
                        saved_outputs[key] = self.layers[key].call_with_external_weights(param, [saved_outputs[v] for v in val])
        if self.multiple_branches:
            return [saved_outputs[layer.name] for layer in self.outputs]
        else:
            return saved_outputs[self.outputs[0].name]

    def __call__(self, inputs):
        return self.call_with_external_weights(self.params, inputs)
