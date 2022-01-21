from collections import namedtuple, deque
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
        
        for arg, input_layer in zip(args, self.inputs):
            input_layer(arg)

        for layer in self.layers:
            prevs = layer._node_container.inbound_nodes
            if not isinstance(layer, Input):
                if len(prevs) > 1:
                    outputs = layer([prev.output for prev in prevs])
                elif len(prevs) == 1:
                    outputs = layer(prevs[0].output)
        return outputs

    def call_with_external_weights(self, params, *args):
        for arg, input_layer in zip(args, self.inputs):
            input_layer(arg)

        for layer, param in zip(self.layers, params):
            prevs = layer._node_container.inbound_nodes
            if not isinstance(layer, Input):
                if len(prevs) > 1:
                    outputs = layer.call_with_external_weights(param, [prev.output for prev in prevs])
                elif len(prevs) == 1:
                    outputs = layer.call_with_external_weights(param, prevs[0].output)
        return outputs
    
    def __call__(self, inputs):
        return self.call(inputs)

    def _create_call_function(self):
        pass