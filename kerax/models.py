import set_path
from layers import core
from layers import convolutional as c
from jax.experimental import optimizers
import numpy as np
from jax import value_and_grad

class Model:
    def __init__(self, input_layer, output_layer, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.name = name if name is not None else self.__class__.__name__
        self.get_layers_and_params()
        self.layers = self.layers[::-1]
        self.trainable_params = self.trainable_params[::-1]
        self.predict = self.__call__
    
    def get_layers_and_params(self):
        pointer = self.output_layer
        self.layers = []
        self.trainable_params = []
        while hasattr(pointer, 'prev'):
            self.layers.append(pointer)
            self.trainable_params.append(pointer.get_weights())
            pointer = pointer.prev

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def __call__(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

    def call_with_external_weights(self, x, params):
        for i, layer in enumerate(self.layers):
            x = layer.call_with_external_weights(x, params[i])
        return x
    
        



###Testing
inputs = core.Input((64,28,28,1))
conv1 = c.Conv2D(3,3)(inputs)
conv2 = c.Conv2D(3,3)(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512)(flatten)
output = core.Dense(10)(dense)
output_activation = core.Activation('softmax')(output)

model = Model(inputs, output_activation)

x = np.random.random((64,28,28,1))
y = np.random.random((64,10))

print(model.predict(x))

def loss(params, x, y):
    """ Compute the multi-class cross-entropy loss """
    preds = model.call_with_external_weights(x, params)
    return -np.sum(preds * y)


def update(params, x, y, opt_state):
    """ Compute the gradient for a batch and update the parameters """
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value

step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(model.trainable_params)


new_params, opt_state, val = update(get_params(opt_state), x, y, opt_state )

print(np.any(new_params[0][0] == model.trainable_params[0][0]))
