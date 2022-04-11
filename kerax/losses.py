from enum import Enum
from typing import Union, NamedTuple
from jax import lax
from jax import numpy as jnp
import optax  # type:ignore
from . import backend
from jax import jit


class REDUCTION(Enum):
    SUM = "sum"
    MEAN = "mean"
    AUTO = "auto"


class Reducer:
    def __init__(self, reduction=None):
        self.reduction = reduction
        if reduction == "mean":
            self.reduce = self.reduce_by_mean
        elif reduction == "sum":
            self.reduce = self.reduce_by_sum
        else:
            self.reduce = lambda inputs: inputs

    def reduce_by_mean(self, inputs):
        return jnp.mean(inputs)

    def reduce_by_sum(self, inputs):
        return jnp.sum(inputs)

    def __call__(self, inputs):
        return self.reduce(inputs)


class LossOutputs(NamedTuple):
    loss: jnp.DeviceArray
    predictions: jnp.DeviceArray


class Loss:
    def __init__(self, reduction=None, name=None):
        self.reduction = Reducer(reduction)
        self.name = self.__class__.__name__ if name is None else name
        self.epsilon = 1e-12

    def setup_loss(self, model):
        self.model = model
        if backend.is_jit_enabled():
            self.call = jit(self.call)

    @property
    def __name__(self):
        return self.name

    def call(self, params, x, y):
        raise NotImplementedError("Must be implemented in subclass")

    def __call__(self, params, train_x, y_true):
        return self.call(params, train_x, y_true)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class CategoricalCrossEntropy(Loss):
    def __init__(self, with_logits=False, reduction=None, name=None):
        super(CategoricalCrossEntropy, self).__init__(reduction, name)
        self.with_logits = with_logits
        if with_logits:
            self.call = self._call_with_logits
        else:
            self.call = self._call_with_probabilities

    def _call_with_logits(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        loss = jnp.sum(optax.softmax_cross_entropy(y_preds, y)) / y_preds.shape[0]
        return LossOutputs(loss, y_preds)

    def _call_with_probabilities(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        y_preds = jnp.clip(y_preds, self.epsilon, 1.0 - self.epsilon)
        loss = -jnp.sum(y * jnp.log(y_preds + 1e-9)) / y_preds.shape[0]
        return LossOutputs(loss, y_preds)


class MeanSquaredError(Loss):
    def __init__(self, reduction=None, name=None):
        super(MeanSquaredError, self).__init__(reduction, name)

    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        loss = jnp.mean(jnp.square(jnp.subtract(y_preds, y)))
        return LossOutputs(loss, y_preds)


class MeanAbsoluteError(Loss):
    def __init__(self, reduction=None, name=None):
        super(MeanAbsoluteError, self).__init__(reduction, name)

    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        loss = jnp.mean(jnp.abs(jnp.subtract(y_preds, y)))
        return LossOutputs(loss, y_preds)


class Huber(Loss):
    def __init__(self, reduction, delta=1.0, name=None):
        super(Huber, self).__init__(reduction, name)
        self.delta = delta

    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        loss = optax.huber_loss(y_preds, y)
        loss = jnp.sum(loss) / y_preds.shape[0]
        return LossOutputs(loss, y_preds)


class BinaryCrossEntropy(Loss):
    def __init__(self, reduction=None, name=None):
        super(BinaryCrossEntropy, self).__init__(reduction, name)

    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        lhs = y * jnp.log(y_preds * self.epsilon)
        rhs = (1 - y) * jnp.log(1 - y_preds + self.epsilon)
        loss = -jnp.mean(lhs + rhs)
        return LossOutputs(loss, y_preds)


class CosineDistance(Loss):
    def __init__(self, reduction=None, name=None):
        super(CosineDistance, self).__init__(reduction=reduction, name=name)

    def call(self, params, x, y):
        y_preds = self.model.call_with_external_weights(params, x)
        loss = optax.cosine_distance(y_preds, y)
        loss = jnp.sum(loss) / y_preds.shape[0]
        return LossOutputs(loss=loss, predictions=y_preds)


supported_losses = {
    "binary_crossentropy": BinaryCrossEntropy,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "cosine_distance": CosineDistance,
}


def get(identifier: Union[str, Loss]) -> Loss:
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
