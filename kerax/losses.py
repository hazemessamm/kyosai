from jax import lax
from jax import numpy as jnp  # type:ignore


class Reducer:
    def __init__(self, reduction=None):
        self.reduction = reduction
        if reduction == 'mean':
            self.reduce = self.reduce_by_mean
        elif reduction == 'sum':
            self.reduce = self.reduce_by_sum
        else:
            self.reduce = lambda inputs: inputs

    def reduce_by_mean(self, inputs):
        return jnp.mean(inputs)
    
    def reduce_by_sum(self, inputs):
        return jnp.sum(inputs)

    def __call__(self, inputs):
        return self.reduce(inputs)

class Loss:
    def __init__(self, model, reduction=None, name=None):
        self.reduction = Reducer(reduction)
        self.name = self.__class__.__name__ if name is None else name
        self.epsilon = 1e-12
        self.model = model
    
    @property
    def __name__(self):
        return self.name

    def call(self, params, x, y):
        raise NotImplementedError("Must be implemented in subclass")
    
    def __call__(self, params, train_x, y_true):
        return self.call(params, train_x, y_true)
    
    def get_config(self):
        return {'reduction': self.reduction, 'name': self.name}

class CategoricalCrossEntropy(Loss):
    def __init__(self, model, reduction=None, name=None):
        super(CategoricalCrossEntropy, self).__init__(model, reduction, name)
    
    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        y_preds = jnp.clip(y_preds, self.epsilon, 1. - self.epsilon)
        num_samples = y_preds.shape[0]
        return -jnp.sum(y * jnp.log(y_preds + 1e-9)) / num_samples


class MeanSquaredError(Loss):
    def __init__(self, model, reduction=None, name=None):
        super(MeanSquaredError, self).__init__(model, reduction, name)
    
    def call(self, params, x, y):
        y_pred = self.model.call_with_external_weights(params, x)
        return jnp.mean(jnp.square(jnp.subtract(y_pred, y)))
    
class MeanAbsoluteError(Loss):
    def __init__(self, model, reduction=None, name=None):
        super(MeanAbsoluteError, self).__init__(model, reduction, name)

    def call(self, params, x, y):
        y_pred = self.model.call_with_external_weights(params, x)
        return jnp.mean(jnp.abs(jnp.subtract(y_pred, y)))
    

class Huber(Loss):
    def __init__(self, model, reduction, delta=1.0, name=None):
        super(Huber, self).__init__(model, reduction, name)
        self.delta = delta
    
    def call(self, params, x, y):
        y_pred = self.model.call_with_external_weights(params, x)
        error = jnp.subtract(y_pred, y)
        abs_error = jnp.abs(error)
        half = jnp.array(0.5, dtype=abs_error.dtype)
        return jnp.mean(jnp.where(abs_error <= self.delta, 
        half * jnp.square(error), self.delta*abs_error-half*jnp.square(self.delta)), axis=-1)


class BinaryCrossEntropy(Loss):
    def __init__(self, model, reduction=None, name=None):
        super(BinaryCrossEntropy, self).__init__(model, reduction, name)
    
    def call(self, params, x, y):
        y_pred = self.model.call_with_external_weights(params, x)
        lhs = y * jnp.log(y_pred *self.epsilon)
        rhs = (1 - y) * jnp.log(1-y_pred+self.epsilon)
        return -jnp.mean(lhs + rhs)
    

supported_losses = {
    'binary_crossentropy': BinaryCrossEntropy,
    'categorical_crossentropy': CategoricalCrossEntropy,
    'mse': MeanSquaredError,
    'mean_squared_error': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'mean_absolute_error': MeanAbsoluteError,
}

def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        loss_fn = supported_losses.get(identifier, None)
        if loss_fn is None:
            raise Exception("Cannot find the specified loss function")
        else:
            return loss_fn
    elif isinstance(identifier, Loss):
        return identifier
