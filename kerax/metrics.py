from jax import numpy as jnp
from kerax.engine import Trackable
from . import utils
from . import losses


class Metric(Trackable):
    def __init__(self, name=None):
        self.name = name

    def __call__(self, x, y):
        raise NotImplementedError("Should be implemented in a subclass")


class Accuracy(Metric):
    def __init__(self, name=None):
        super(Metric, self).__init__(name)

    def __call__(self, y_true, y_pred):
        if y_true.shape[-1] > 1:
            y_true = utils.to_numbers(y_true)
        if y_pred.shape[-1] > 1:
            y_pred = utils.to_numbers(y_pred)
        accuracy = jnp.sum(y_true == y_pred)
        return accuracy / y_true.shape[0]


supported_metrics = {"accuracy": Accuracy}


def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        return supported_metrics.get(identifier, None)
    else:
        raise Exception("Cannot find the specified identifier")
