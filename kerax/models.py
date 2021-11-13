from __future__ import absolute_import
from kerax import optimizers
from jax import numpy as jnp #type: ignore
from kerax import layers
from kerax import losses
from tqdm import tqdm, trange
from kerax.engine.data_adapter import TensorLikeDataAdapter #type: ignore
from kerax.engine.graph import Graph
from kerax.engine.trackable import Trackable
from kerax.layers.core import Layer



class Model:
    '''
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    '''
    SUPPORTED_KWARGS = {
            'input',
            'output',
            'inputs',
            'outputs',
            'name'
        }
    def __init__(self, *args, **kwargs):        
        self.name = kwargs.pop('name', self.__class__.__name__)
        #Aliases
        self.predict = self.__call__
        self.predict_with_external_weights = self.call_with_external_weights
        self._built = False
        self.trainable = kwargs.get('trainable', True)
        self._training_phase = False
        self._compiled = False
        self._sequential_model = isinstance(self, Sequential)
        self.initialize_graph(*args, **kwargs)

    @property
    def layers(self):
        if self._sequential_model:
            return self._layers
        return self.graph.layers

    @property
    def params(self):
        if self._sequential_model:
            return self._params
        return self.graph.params
    
    @property
    def weights(self):
        if self._sequential_model:
            return self._params
        return self.graph.params

    @property
    def compiled(self):
        return self._compiled

    def initialize_graph(self, *args, **kwargs):
        'Stores the layers and paramters'
        if not self._sequential_model:
            self.graph = Graph(**kwargs)

    def compile(self, loss, optimizer, metrics=['loss']):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = losses.get(loss)(self)
        self.optimizer = optimizers.get(optimizer)
        self.metrics = metrics
        #if optimizer is string then it needs configuration
        if isinstance(optimizer, str):
            self.optimizer = self.optimizer(loss_fn=self.loss_fn, model=self)
        self._compiled = True


    def __call__(self, x, training=False):
        'Takes inputs and returns predictions'
        return self.graph(x)

    def call_with_external_weights(self, params, x, training=False):
        'Takes inputs and params and returns predictions'
        return self.graph.call_with_external_weights(params, x)

    def get_weights(self):
        if self._sequential_model:
            return self._params
        return self.graph.params

    def set_weights(self, weights):
        'Set new weights on every layer'
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self.graph.params = weights

    def update_weights(self, weights):
        self.graph.params.clear()
        for layer, w in zip(self.layers, weights):
            layer.update_weights(w)
            self.graph.params.append(layer.params)

    def train_step(self, x, y):
        'Returns loss value and takes training batch'
        loss = self.loss_fn(self.graph.params, x, y)
        grads = self.optimizer.get_gradients(x, y)
        self.optimizer.apply_gradients(grads)
        return loss

    def test_step(self, x, y):
        batch_x, batch_y = self.input.get_validation_batch()
        return self.loss_fn(self.graph.params, batch_x, batch_y)

    def fit(self, x, y, epochs=1, batch_size=None, steps=None, shuffle=True, validation_data=None):
        #if the model is not compiled then it will raise exception
        if not self.compiled:
            raise Exception('Model is not compiled, use compile() method')

        self.dataset = TensorLikeDataAdapter(x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=shuffle)

        for epoch in range(1, epochs+1):
            remaining_epochs = int(epochs - epoch)
            #gets a batch and pass it to the model
            with trange(len(self.dataset), bar_format='{l_bar}{bar:40}{r_bar}{bar:-20b}') as t:
                for _ in t:
                    batch_x, batch_y = self.dataset.get_batch()
                    train_loss = self.train_step(batch_x, batch_y)
                    if validation_data:
                        validation_loss = self.test_step(validation_data[0], validation_data[1])
                        t.set_postfix(loss=train_loss, validation_loss=validation_loss, remaining_epochs=remaining_epochs)
                    else:
                        t.set_postfix(loss=train_loss, remaining_epochs=remaining_epochs)
                    t.refresh()


class Sequential(Model):
    def __init__(self, layers=None, **kwargs):
        super().__init__(**kwargs)
        
        self._layers = layers if layers is not None else []
        self._params = []
        self.validate_init()
    
    def validate_init(self):
        if self._layers is not None:
            self.connect_layers()

    def connect_layers(self):
        for i in range(len(self.layers)-1):
            self._layers[i+1](self.layers[i])

    def add(self, layer):
        if isinstance(layer, Layer):
            if len(self._layers) >= 1:
                layer(self._layers[-1])
            self._layers.append(layer)
            self.params.append(layer.params)
        else:
            raise Exception('add() only accepts layers subclass instances or Input instance')

    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def call_with_external_weights(self, params, inputs):
        'Takes inputs and params and returns predictions'
        outputs = inputs
        for layer in self.layers:
            outputs = layer.call_with_external_weights(params, outputs)
        return outputs

    def set_weights(self, weights):
        'Set new weights on every layer'
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self._params = weights

    def update_weights(self, weights):
        self._params.clear()
        for layer, w in zip(self.layers, weights):
                layer.update_weights(w)
                self._params.append(layer.params)
    