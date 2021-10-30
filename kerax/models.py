from __future__ import absolute_import
import optimizers
from jax import numpy as jnp #type: ignore
import layers
import losses
from tqdm import tqdm, trange #type: ignore






class Model:
    '''
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    '''
    supported_kwargs = {
            'input',
            'output',
            'inputs',
            'outputs',
            'name',
            'trainable'
        }
    def __init__(self, *args, **kwargs):
        self.input = kwargs.get('input')
        self.output = kwargs.get('output')
        self.name = kwargs.get('name', self.__class__.__name__)
        #Aliases
        self.predict = self.__call__
        self.predict_with_external_weights = self.call_with_external_weights
        #list to store the layers
        self.layers = []
        #list to store the parameters
        self.params = []
        self.built = False
        self.trainable = kwargs.get('trainable', True)
        self.training_phase = False
        self.compiled = False

        self.initialize_graph()
    
    def initialize_graph(self):
        'Stores the layers and paramters'

        #temporary variable to loop over the layers
        pointer = self.output

        #looping over the layers
        while hasattr(pointer, 'prev'):
            self.layers.insert(0, pointer)
            self.params.insert(0, pointer.get_weights())
            pointer = pointer.prev
        self.built = True
        

    def compile(self, loss, optimizer, metrics=['loss']):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = losses.get(loss)(self)
        self.optimizer = optimizers.get(optimizer)
        self.metrics = metrics

        #if optimizer is string then it needs configuration
        if isinstance(optimizer, str):
            self._configure_optimizer()
        self.compiled = True

    
    def _configure_optimizer(self):
        'Configure the optimizer'
        self.optimizer = self.optimizer(loss_fn=self.loss_fn, model=self)

    def __call__(self, x, training=False):
        'Takes inputs and returns predictions'
        return self.call_with_external_weights(self.params, x)

    def call_with_external_weights(self, params, x, training=False):
        'Takes inputs and params and returns predictions'
        for layer, param in zip(self.layers, params):
            x = layer.call_with_external_weights(param, x)
        return x

    def get_weights(self):
        return self.params

    def set_weights(self, weights):
        'Set new weights on every layer'
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self.params = weights

    def update_weights(self, weights):
        self.params.clear()
        for layer, w in zip(self.layers, weights):
                layer.update_weights(w)
                self.params.append(layer.params)

    def train_step(self, x, y, **kwargs):
        'Returns loss value and takes training batch'
        loss = self.loss_fn(self.params, x, y)
        grads = self.optimizer.get_gradients(x, y)
        self.optimizer.apply_gradients(grads)
        return loss

    def test_step(self, x, y):
        batch_x, batch_y = self.input.get_validation_batch()
        return self.loss_fn(self.params, batch_x, batch_y)


    def fit(self, x, y, epochs=1, batch_size=1, validation_data=None):
        #if the model is not compiled then it will raise exception
        if not self.compiled:
            raise Exception(f'Model is not compiled, use compile() method')
        
        #sets the batch size
        self.input.set_batch_size(batch_size)
        #stores the data to the input layer and validation data if there is validation data
        self.input.store_data(x, y, validation_data=validation_data)

        for epoch in range(epochs):
            remaining_epochs = int(epochs - epoch)
            #gets a batch and pass it to the model
            with trange(self.input.num_batches, bar_format='{l_bar}{bar:40}{r_bar}{bar:-20b}') as t:
                for _ in t:
                    batch_x, batch_y = self.input.get_training_batch()
                    train_loss = self.train_step(batch_x, batch_y, validation_data=validation_data)
                    if validation_data:
                        validation_loss = self.test_step(validation_data[0], validation_data[1])
                        t.set_postfix(loss=train_loss, validation_loss=validation_loss, remaining_epochs=remaining_epochs)
                    else:
                        t.set_postfix(loss=train_loss, remaining_epochs=remaining_epochs)
                    t.refresh()


class Sequential:
    def __init__(self, layers=None, trainable=True, name=None):
        self.layers = layers
        self.trainable = trainable
        self.name = name
        self.params = []
        self.validate_init()
    
    def validate_init(self):
        if self.layers is not None:
            self.connect_layers()
        
        if self.name is None:
            self.name = self.__class__.__name__ 

    def connect_layers(self):
        current_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer(current_layer)
            
            if hasattr(current_layer, 'params'):
                self.params.append(current_layer.params)
            
            current_layer = layer
            
    
    def add(self, layer):
        if isinstance(layer, (layers.Input, layers.Layer)):
            self.temporary_layers_list.append(layer)
        else:
            raise Exception('add() only accepts layers subclass instances or Input instance')
    
    def compile(self, loss, optimizer, metrics=['loss', 'remaining epochs']):
        self.loss_fn = loss
        self.optimizer = optimizers.get(optimizer)
        self.metrics = metrics

        if not self.built:
            connected_layers = self.connect_layers(self.temporary_layers_list)
            self.input_layer = connected_layers[0]
            self.output_layer = connected_layers[-1]
            self.get_layers_and_params()
            self.temporary_layers_list = None
            
        if isinstance(optimizer, str):
            self.configure_optimizer()