from __future__ import absolute_import
import set_path
from optimizers import optimizers
import sys
from jax import numpy as jnp
from utils import Progbar

class Model:
    '''
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    '''
    def __init__(self, input_layer, output_layer, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.name = name if name is not None else self.__class__.__name__
        self.get_layers_and_params()
        self.layers = self.layers[::-1]
        self.trainable_params = self.trainable_params[::-1]
        self.predict = self.__call__
        self.predict_with_external_weights = self.call_with_external_weights
 
    
    def get_layers_and_params(self):
        'Stores the layers and paramters'

        #temporary variable to loop over the layers
        pointer = self.output_layer

        #list to store the layers
        self.layers = []

        #list to store the parameters
        self.trainable_params = []

        #looping over the layers
        while hasattr(pointer, 'prev'):
            self.layers.append(pointer)
            self.trainable_params.append(pointer.get_weights())
            pointer = pointer.prev

    def compile(self, loss, optimizer, metrics=['Loss', 'Remaining epochs']):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = loss
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
            self.metrics += ['Validation loss']
        
        prgbar = Progbar(self.input_layer.num_batches, stateful_metrics=self.metrics)


        for epoch in range(epochs):
            finished_batches = 0
            #gets a batch and pass it to the model
            for _ in range(self.input_layer.num_batches):
                batch_x, batch_y = self.input_layer.get_training_batch()
                loss = self.train_step(batch_x, batch_y, validation_data=validation_data)                
                values = None
                if isinstance(loss, tuple):
                    values = [('Remaining epochs', epochs-epoch), ('Loss', loss[0]), ('Validation loss', loss[1])]
                else:
                    values = [('Remaining epochs', epochs-epoch), ('Loss', loss)]
                finished_batches += 1
                prgbar.update(finished_batches, values=values)

            if validation_data is not None:
                print(f"Training loss: {jnp.mean(epoch_training_loss)}, Validation loss: {jnp.mean(epoch_validation_loss)}")
            else:
                print(f"Training loss: {epoch_training_loss}")
                
            


