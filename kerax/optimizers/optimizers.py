from __future__ import absolute_import
import set_path
from jax.experimental import optimizers
from jax import value_and_grad, grad

class Optimizer:
    def __init__(self, loss_fn, model, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.model = model
        self.step_index = 0

    def apply_grads(self, grads):
        raise NotImplementedError

    def step(self, x, y):
        value, grads = self.get_value_and_gradients(x, y)
        self.apply_grads(grads)
        return value
    
    def increment_step_index(self):
        self.step_index += 1

    def get_value_and_gradients(self, x, y):
        value, grads = value_and_grad(self.loss_fn)(self.model.trainable_params, x, y)
        return value, grads



class SGD(Optimizer):
    def __init__(self, loss_fn, model, learning_rate=0.001):
        super(SGD, self).__init__(loss_fn=loss_fn, model=model, learning_rate=0.001)
        self.init_fn, self.update_fn, self.get_params = optimizers.sgd(learning_rate)
        self.optimizer_state = self.init_fn(model.trainable_params)

    def apply_grads(self, grads, step=0):
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        self.model.set_weights(self.get_params(self.optimizer_state))


class Adam(Optimizer):
    def __init__(self, loss_fn, model, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super(Adam, self).__init__(loss_fn=loss_fn, model=model, learning_rate=0.001)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.init_fn, self.update_fn, self.get_params = optimizers.sgd(learning_rate)
        self.optimizer_state = self.init_fn(model.trainable_params)

    def apply_grads(self, grads, step=0):
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        self.model.set_weights(self.get_params(self.optimizer_state))


class Adagrad(Optimizer):
    def __init__(self, loss_fn, model, learning_rate=0.001, momentum=0.9):
        super(Adagrad, self).__init__(loss_fn=loss_fn, model=model, learning_rate=learning_rate)
        self.momentum = momentum
        self.init_fn, self.update_fn, self.get_params = optimizers.sgd(learning_rate)
        self.optimizer_state = self.init_fn(model.trainable_params)
    
    def apply_grads(self, grads, step=0):
        self.optimizer_state = self.update_fn(self.step_index, grads, self.optimizer_state)
        self.model.set_weights(self.get_params(self.optimizer_state))
    


    
opts = {
    'sgd': SGD,
    'adam': Adam,
    'adagrad': Adagrad
}

def get(identifier):
    if identifier is None:
        return None
    elif isinstance(identifier, Optimizer):
        return identifier
    else:
        optimizer = opts.get(identifier, None)
        if optimizer is None:
            raise Exception('Cannot find the specified optimizer')
        return optimizer
