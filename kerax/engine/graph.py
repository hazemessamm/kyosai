from collections import deque, namedtuple, OrderedDict

from jax import numpy as jnp
from kerax.layers.core import Input


class Graph:
    def __init__(self, **kwargs):
        self._validate_init(**kwargs)
        self.connected_layers = []
        self.connection = namedtuple('Layer', ['layer1', 'layer2'])
        self.layers = []
        self.params = []
        self.get_layers()

    def flatten(self, x):
        def _flatten(x, result=[]):
            for i in x:
                if isinstance(i, list):
                    return _flatten(i, result)
                else:
                    result.append(i)
            return result
        return _flatten(x, [])

    def _validate_init(self, **kwargs):
        if (
            kwargs.get('inputs', False)
            and isinstance(kwargs.get('inputs', False), (list, tuple)) != 1
        ):
            raise Exception('Use \'input\' argument instead of \'inputs\' if you want to pass a list or a tuple')
        elif (
            kwargs.get('input', False)
            and isinstance(kwargs.get('input', False), (list, tuple)) >= 1
        ):
            raise Exception('Use \'inputs\' argument instead of \'input\' if you want to pass an input layer')

        inputs = kwargs.get('inputs', False) or kwargs.get('input', False)

        if not inputs:
            raise Exception('inputs should be provided')

        if isinstance(inputs, (list, tuple)):
            self.inputs = self.flatten(inputs)
            self.input = None
        else:
            self.input = inputs
            self.inputs = [inputs]

        outputs = kwargs.get('outputs', False) or kwargs.get('output', False)

        if not outputs:
            raise Exception('outputs should be provided')

        if (
            kwargs.get('outputs', False)
            and isinstance(kwargs.get('outputs', False), (list, tuple)) != 1
        ):
            raise Exception('Use \'output\' argument instead of \'outputs\' if you want to pass a list or a tuple')
        elif (
            kwargs.get('output', False)
            and isinstance(kwargs.get('output', False), (list, tuple)) >= 1
        ):
            raise Exception('Use \'outputs\' argument instead of \'output\' if an output layer')

        if isinstance(outputs, (list, tuple)):
            self.outputs = self.flatten(outputs)
            self.output = None
        else:
            self.output = outputs
            self.outputs = [outputs]

    def get_layers(self):
        queue = deque()

        if self.inputs:
            queue += self.flatten(self.inputs)
        else:
            queue.append(self.input)

        if self.outputs:
            queue += self.flatten(self.outputs)
        else:
            queue.append(self.output)

        visited = {i.depth for i in queue}
        self.layers = [*queue]

        while queue:
            current_pointer = queue.popleft()
            for layer in current_pointer._node_container.outbound_nodes:
                if layer.depth not in visited:
                    self.layers.append(layer)
                    queue.append(layer)
                    visited.add(layer.depth)

        self.layers = sorted(self.layers, key=lambda layer: layer.depth)
        
        for layer in self.layers:
            self.params.append(layer.params)

    def connect_layers(self):
        self.get_layers()
        self.layers = sorted(self.layers, key=lambda x: x.depth)

        for layer in self.layers:
            self.connected_layers += [self.connection(layer, i) for i in layer.next]
    
    def same_input_len(self, inputs):
        return len(inputs) == len(self.inputs)

    def call(self, *args):
        'This method is responsible for flowing the data through the graph (Functional Model)'
        if not self.same_input_len(args):
            raise Exception(f'Not the same input length expected {len(self.inputs)} found {len(args)}')
        

        consumed_args = 0
        outs = []
        for layer in self.layers:
            prevs = layer._node_container.inbound_nodes
            if not prevs and isinstance(layer, Input):
                layer(args[consumed_args])
                consumed_args += 1
            elif len(prevs) > 1:
                prev_outs = jnp.array([prev.get_output() for prev in prevs])
                out = layer(prev_outs)
            elif len(prevs) == 1:
                out = layer(prevs[0].get_output())
            
            if layer in self.outputs:
                outs.append(out)
        return outs if len(outs) > 1 else outs[0]


    def call_with_external_weights(self, params, *args):
        if len(args) != len(self.inputs):
            raise Exception(f'Not the same input length expected {len(self.inputs)} found {len(args)}')
        
        consumed_args = 0
        outs = []
        for param, layer in zip(params, self.layers):
            prevs = layer._node_container.inbound_nodes
            if not prevs and isinstance(layer, Input):
                layer.call_with_external_weights(param, args[consumed_args])
                consumed_args += 1
            elif len(prevs) > 1:
                prev_outs = jnp.array([prev.get_output() for prev in prevs])
                out = layer.call_with_external_weights(param, prev_outs)
            elif len(prevs) == 1:
                out = layer.call_with_external_weights(param, prevs[0].get_output())

            if layer in self.outputs:
                outs.append(out)

        return outs if len(outs) > 1 else outs[0]
    
    def __call__(self, inputs):
        return self.call(inputs)

    def _create_call_function(self):
        pass


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
