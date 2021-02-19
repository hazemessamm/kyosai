from __future__ import absolute_import
import set_path
from optimizers import optimizers
import sys
from jax import numpy as jnp

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

    def compile(self, loss, optimizer, record_loss=False):
        'Takes the loss, optimizer and loss recorder state'
        self.loss_fn = loss
        self.optimizer = optimizers.get(optimizer)

        #if optimizer is string then it needs configuration
        if isinstance(optimizer, str):
            self.configure_optimizer()
        if record_loss:
            self.loss_history = []
    
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
            validation_loss = self.loss_fn(model.trainable_params, batch_x, batch_y)
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


        for epoch in range(epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0 
            #gets a batch and pass it to the model
            batch_x, batch_y = self.input_layer()
            for _ in range(self.input_layer.num_batches):
                loss = self.train_step(batch_x, batch_y, validation_data=validation_data)
                sys.stdout.write(f"Progress {self.input_layer.training_index}/{self.input_layer.num_batches}\r")
                sys.stdout.flush()
                if isinstance(loss, tuple):
                    epoch_training_loss += loss[0]
                    epoch_validation_loss += loss[1]
                else:
                    epoch_training_loss += loss
                batch_x, batch_y = self.input_layer()

            if validation_data is not None:
                print(f"Training loss: {jnp.mean(epoch_training_loss)}, Validation loss: {jnp.mean(epoch_validation_loss)}")
            else:
                print(f"Training loss: {epoch_training_loss}")
                
            


