from jax import numpy as jnp


def CategoricalCrossEntropy(model, epsilon=1e7):
    def _CategoricalCrossEntropy(params, x, y):
        y_pred = model.call_with_external_weights(x, params)
        return jnp.sum(-y*jnp.log(y_pred+epsilon))
    return _CategoricalCrossEntropy
    

def MeanSquaredError(model):
    def _MeanSquaredError(params, x, y):
        y_pred = model.call_with_external_weights(x, params)
        return jnp.mean(jnp.power(y_pred-y, 2))
    return _MeanSquaredError
    

def MeanAbsoluteError(model):
    def _MeanAbsoluteError(params, x, y):
        y_pred = model.call_with_external_weights(x, params)
        return jnp.mean(jnp.abs(y_pred-y))
    return _MeanAbsoluteError
    

def NegativeLogLikelihood(model, epsilon=1e7):
    def _NegativeLogLikelihood(params, x, y):
        y_pred = model.call_with_external_weights(x, params)
        return jnp.mean(-jnp.log(y_pred[y] + epsilon))
    return _NegativeLogLikelihood


def Huber(params, x, y):
    pass

def CosineSimilarity(params, x, y):
    pass


def BinaryCrossEntropy(model, epsilon=1e7):
    def _BinaryCrossEntropy(params, x, y):
        y_pred = model.call_with_external_weights(x, params)
        lhs = y * jnp.log(y_pred  * epsilon)
        rhs = (1 - y) * jnp.log(1-y_pred+epsilon)
        return -jnp.mean(lhs + rhs)
    return _BinaryCrossEntropy
    


losses_dict = {
    'binary_crossentropy': BinaryCrossEntropy,
    'negative_log_likelihood': NegativeLogLikelihood,
    'categorical_crossentropy': CategoricalCrossEntropy,
    'mse': MeanSquaredError,
    'mean_squared_error': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'mean_absolute_error': MeanAbsoluteError
}

def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        loss_fn = losses_dict.get(identifier, None)
        if loss_fn is None:
            raise Exception("Cannot find the specified loss function")
        else:
            return loss_fn
        
