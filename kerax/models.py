from __future__ import absolute_import
import set_path
from layers import core
from layers import convolutional as c
import numpy as np
from optimizers import optimizers

class Model:
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
        pointer = self.output_layer
        self.layers = []
        self.trainable_params = []
        while hasattr(pointer, 'prev'):
            self.layers.append(pointer)
            self.trainable_params.append(pointer.get_weights())
            pointer = pointer.prev

    def compile(self, loss, optimizer, record_loss=False):
        self.loss_fn = loss
        self.optimizer = optimizers.get(optimizer)
        if isinstance(optimizer, str):
            self.configure_optimizer()
        if record_loss:
            self.loss_history = []
    
    def configure_optimizer(self):
        self.optimizer = self.optimizer(loss_fn=self.loss_fn, model=self)

    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def call_with_external_weights(self, x, params):
        for i, layer in enumerate(self.layers):
            x = layer.call_with_external_weights(x, params[i])
        return x

    def set_weights(self, weights):
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self.trainable_params = weights

    def train_step(self, x, y):
        loss = self.optimizer.step(x, y)
        return loss

    def fit(self, x, y, epochs=1, batch_size=1, validation_data=None):
        if not hasattr(self, 'loss_fn') or not hasattr(self, 'optimizer'):
            raise Exception(f'Model is not compiled, found loss={None} and optimizer={None}')
        
        self.input_layer.set_batch_size(batch_size)
        self.input_layer.store_data(x, y, validation_data=validation_data)
        for epoch in range(epochs):
            batch_x, batch_y = self.input_layer()
            loss = self.train_step(x, y)
            print(f"Training Loss: {loss}")
            if validation_data is not None:
                batch_x, batch_y = self.input_layer.get_validation_batch()
                loss = self.loss_fn(model.trainable_params, batch_x, batch_y)
                print(f'Validation Loss: {loss}')



###Testing
inputs = core.Input((64,28,28,1))
conv1 = c.Conv2D(128,3, activation='relu')(inputs)
conv2 = c.Conv2D(64,3, activation='relu')(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512, activation='relu')(flatten)
output = core.Dense(10)(dense)
output_activation = core.Activation('softmax')(output)

model = Model(inputs, output_activation)


def loss(params, x, y):
    preds = model.call_with_external_weights(x, params)
    return -np.sum(preds * y)

sgd = optimizers.Adam(loss_fn=loss, model=model)

model.compile(loss=loss, optimizer='sgd')


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype('float32')
x_test = np.expand_dims(x_test, -1)



y_train = to_categorical(y_train, 10)

model.fit(x_train, y_train, epochs=20, batch_size=64)
