from __future__ import absolute_import
from optimizers import optimizers
import sys
from jax import numpy as jnp
from utils import Progbar
from layers import convolutional as cl
from layers import core
import losses



class Model:
    '''
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    '''
    def __init__(self, input_layer=None, output_layer=None, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.name = name if name is not None else self.__class__.__name__
        #Aliases
        self.predict = self.__call__
        self.predict_with_external_weights = self.call_with_external_weights
        #list to store the layers
        self.layers = []
        #list to store the parameters
        self.trainable_params = []

        self.built = False


        if not isinstance(self, Sequential):
            if input_layer is None and output_layer is None:
                raise Exception('the model should has input_layer and output_layer')
            self.get_layers_and_params()

 
    
    def get_layers_and_params(self):
        'Stores the layers and paramters'

        #temporary variable to loop over the layers
        pointer = self.output_layer

        #looping over the layers
        while hasattr(pointer, 'prev'):
            self.layers.append(pointer)
            self.trainable_params.append(pointer.get_weights())
            pointer = pointer.prev
        self.layers = self.layers[::-1]
        self.trainable_params = self.trainable_params[::-1]
        self.built=True
        

    def compile(self, loss, optimizer, metrics=['loss', 'remaining epochs']):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = losses.get(loss)
        self.optimizer = optimizers.get(optimizer)
        self.metrics = metrics

        #if optimizer is string then it needs configuration
        if isinstance(optimizer, str):
            self.configure_optimizer()

    
    def configure_optimizer(self):
        'Configure the optimizer'
        self.optimizer = self.optimizer(loss_fn=self.loss_fn, model=self)

    def __call__(self, x):
        'Takes inputs and returns predictions'
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def call_with_external_weights(self, x, params):
        'Takes inputs and params and returns predictions'
        for i, layer in enumerate(self.layers):
            x = layer.call_with_external_weights(x, params[i])
        return x

    def get_weights(self):
        return self.trainable_params

    def set_weights(self, weights):
        'Set new weights on every layer'
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self.trainable_params = weights

    def train_step(self, x, y, **kwargs):
        'Returns loss value and takes training batch'
        training_loss = self.optimizer.step(x, y)
        if kwargs.get('validation_data', None) is not None:
            batch_x, batch_y = self.input_layer.get_validation_batch()
            validation_loss = self.loss_fn(self.trainable_params, batch_x, batch_y)
            return training_loss, validation_loss
        return training_loss

    def fit(self, x, y, epochs=1, batch_size=1, validation_data=None):
        #if the model is not compiled then it will raise exception
        if not hasattr(self, 'loss_fn') or not hasattr(self, 'optimizer'):
            raise Exception(f'Model is not compiled, found loss={None} and optimizer={None}')
        
        #sets the batch size
        self.input_layer.set_batch_size(batch_size)
        #stores the data to the input layer and validation data if there is validation data
        self.input_layer.store_data(x, y, validation_data=validation_data)

        if validation_data is not None:
            self.metrics += ['validation loss']
        prgbar = Progbar(self.input_layer.num_batches, stateful_metrics=self.metrics)
        for epoch in range(epochs):
            finished_batches = 0
            #gets a batch and pass it to the model
            for _ in range(self.input_layer.num_batches):
                batch_x, batch_y = self.input_layer.get_training_batch()
                loss = self.train_step(batch_x, batch_y, validation_data=validation_data)                
                values = None
                if isinstance(loss, tuple):
                    values = [('remaining epochs', epochs-epoch), ('loss', loss[0]), ('validation loss', loss[1])]
                else:
                    values = [('remaining epochs', epochs-epoch), ('loss', loss)]
                finished_batches += 1
                prgbar.update(finished_batches, values=values)
                



class Sequential(Model):
    def __init__(self, layers=None):
        super(Sequential, self).__init__()
        if layers is not None:
            connected_layers = self.connect_layers(layers)
            self.input_layer = connected_layers[0]
            self.output_layer = connected_layers[-1]
            self.get_layers_and_params()
        else:
            self.temporary_layers_list = []

    def connect_layers(self, layers):
        for i, layer in enumerate(reversed(layers)):
            if hasattr(layer, 'prev'):
                layer.connect(layers[len(layers)-2-i])
                layers[i](layers[i-1])
                layers[i+1](layers[i])
        return layers
    
    def add(self, layer):
        if isinstance(layer, core.Input) or isinstance(layer, core.Layer):
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


'''
from jax import random
inputs = core.Input((64,28,28,1))
conv1 = cl.Conv2D(128,3, activation='relu', key=random.PRNGKey(1003))(inputs)
conv2 = cl.Conv2D(64,3, activation='relu', key=random.PRNGKey(1003))(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512, activation='relu', key=random.PRNGKey(1003))(flatten)
output = core.Dense(10, activation='softmax', key=random.PRNGKey(1003))(dense)

model = Model(inputs, output)

import numpy as np

x = np.random.random((64, 28,28,1))
y = [np.random.randint(11) for i in range(64)]
import utils 

y = np.array(y)

y = utils.to_categorical(y)

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y)
'''