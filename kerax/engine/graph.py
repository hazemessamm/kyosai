from collections import namedtuple, deque



class Graph:
    def __init__(self, **kwargs):
        self._validate_init(**kwargs)
        self.connected_layers = []
        self.connection = namedtuple('Layer', ['layer1', 'layer2'])
        self.layers = []
        self.visited = set()
        self.queue = deque()
        self.connect_layers()

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
            self.inputs = None

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
            self.outputs = None

    def get_layers(self):
        if self.inputs:
            self.queue += self.flatten(self.inputs)
        else:
            self.queue.append(self.input)

        if self.outputs:
            self.queue += self.flatten(self.outputs)
        else:
            self.queue.append(self.output)

        self.visited.add(i.index for i in self.queue)
        self.layers += self.queue
                
        while self.queue:
            current_pointer = self.queue.popleft()
            for i in current_pointer.next:
                
                if i.index not in self.visited:
                    self.layers.append(i)
                    self.queue.append(i)
                    self.visited.add(i.index)

    def connect_layers(self):
        self.get_layers()
        self.layers = sorted(self.layers, key=lambda x: x.index)
        self.queue = deque([self.layers[0]])
        self.visited = {self.layers[0].index}

        for layer in self.layers:
            self.connected_layers += [self.connection(layer, i) for i in layer.next]
