from __future__ import absolute_import

from jax import numpy as jnp
from sympy import E  # type: ignore
from tqdm import tqdm, trange

from kerax import losses, optimizers
from kerax.engine import Trackable
from kerax.engine.data_adapter import TensorLikeDataAdapter  # type: ignore
from kerax.engine.graph import GraphV2
from kerax.layers.core import Layer


class Model(Trackable):
    '''
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    '''
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(self.__class__.__name__ if kwargs.get('name') is None else kwargs.get('name'))
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
    def __name__(self):
        return self.name

    @property
    def layers(self):
        if self._sequential_model:
            return self._layers
        return self.graph.layers.values()

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
            self.graph = GraphV2(*args, **kwargs)

    def compile(self, loss, optimizer):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = losses.get(loss)(self)
        self.optimizer = optimizers.get(optimizer)
        #if optimizer is string then it needs configuration
        if isinstance(optimizer, str):
            self.optimizer = self.optimizer(loss_fn=self.loss_fn, model=self)
        self._compiled = True


    def __call__(self, x, training=False):
        'Takes inputs and returns predictions'
        return self.graph(x)

    def call_with_external_weights(self, params, x):
        'Takes inputs and params and returns predictions'
        return self.graph.call_with_external_weights(params, [x])

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
        # loss = self.loss_fn(self.graph.params, x, y)
        loss, grads = self.optimizer.get_loss_and_gradients(x, y)
        self.optimizer.apply_gradients(grads)
        return {'Loss': loss}

    def test_step(self, validation_dataset):
        avg_valid_loss = 0
        for _ in range(validation_dataset.num_batches):
            batch_x, batch_y = validation_dataset.get_batch()
            avg_valid_loss += self.loss_fn(self.graph.params, batch_x, batch_y)
        return {'Validation Loss': avg_valid_loss}


    def _test_step(self, validation_dataset):
        if validation_dataset is None:
            return {}
        else:
            return self.test_step(validation_dataset)

    def fit(self, x, y, epochs=1, batch_size=None, steps=None, shuffle=True, validation_data=None):
        # If the model is not compiled then it will raise exception
        if not self.compiled:
            raise Exception('Model is not compiled, use compile() method')

        dataset = TensorLikeDataAdapter(x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=shuffle)

        if validation_data:
            validation_dataset = TensorLikeDataAdapter(x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=False)
        else:
            validation_dataset = None

        progress_bar = trange(len(dataset), bar_format='{l_bar}{bar:40}{r_bar}{bar:-20b}', ascii=' =')
        
        for epoch in range(1, epochs+1):
            metrics = {}
            for _ in progress_bar:
                batch_x, batch_y = dataset.get_batch()
                loss = self.train_step(batch_x, batch_y)
                metrics.update(loss)
                validation_loss = self._test_step(validation_dataset)
                metrics.update(loss)
                progress_bar.set_postfix(**metrics)
            progress_bar.refresh()


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
        for param, layer in zip(params, self.layers):
            outputs = layer.call_with_external_weights(param, outputs)
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
    