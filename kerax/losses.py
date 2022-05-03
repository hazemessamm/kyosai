from typing import NamedTuple, Union

import optax  # type:ignore
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad

from kerax import backend


class Reducer:
    def __init__(self, reduction="auto"):
        self.reduction = reduction
        if reduction == "mean" or reduction == "auto":
            self.reduce = self.reduce_by_mean
        elif reduction == "sum":
            self.reduce = self.reduce_by_sum
        elif reduction == "none" or reduction == None:
            self.reduce = lambda inputs: inputs
        else:
            raise ValueError(
                f'`reduction` can be `"mean"`, `"sum"`, `"none"` or `None`. Recieved: {reduction}'
            )

    def reduce_by_mean(self, inputs):
        return jnp.sum(inputs) / inputs.shape[0]

    def reduce_by_sum(self, inputs):
        return jnp.sum(inputs)

    def __call__(self, inputs):
        return self.reduce(inputs)


class LossOutputs(NamedTuple):
    """
        grad() returns the gradients only and value_and_grad() 
        evaluates the function and returns it's value and also returns it gradients.
        It will be inefficient to return the loss value then call the same function again to return the gradients,
        so This NamedTuple is used to return the loss, predictions (to pass them to the metrics if any) and the gradients (by using value_and_grad).

        The other solution is call the network 2 times, the first one to return the predictions and pass them to the metrics and loss function,
        The second one when calling it to track the ops and return the gradients. This solution is inefficient.
    """

    loss: jnp.DeviceArray
    predictions: jnp.DeviceArray


class Loss:
    def __init__(self, reduction="auto", name=None):
        self.reduction = Reducer(reduction)
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        self.epsilon = 1e-12
        self.loss_grad_fn = jit(value_and_grad(self._call, 0, has_aux=True))
        self.prediction_fn = None

    @property
    def __name__(self):
        return self.name

    def call(self, params, x, y):
        raise NotImplementedError("Must be implemented in subclass")

    def _call(self, params, train_x, train_y):
        y_preds = self.prediction_fn(params, train_x)
        loss = self.call(train_y, y_preds)
        return LossOutputs(loss=loss, predictions=y_preds)

    def __call__(self, params, train_x, y_true, return_gradients=True):
        if not return_gradients:
            return self._call(params, train_x, y_true)
        else:
            return self.loss_grad_fn(params, train_x, y_true)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class CategoricalCrossEntropyWithLogits(Loss):
    def __init__(self, reduction="auto", name=None):
        super(CategoricalCrossEntropyWithLogits, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        loss = jnp.sum(optax.softmax_cross_entropy(y_preds, y_true)) / y_preds.shape[0]
        return loss


class CategoricalCrossEntropy(Loss):
    def __new__(cls, *args, **kwargs):
        from_logits = kwargs.pop("from_logits", False)
        if from_logits:
            return CategoricalCrossEntropyWithLogits(*args, **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, from_logits=False, reduction="auto", name=None):
        super(CategoricalCrossEntropy, self).__init__(reduction, name)
    def call(self, y_true, y_preds):
        y_preds = jnp.clip(y_preds, self.epsilon, 1.0 - self.epsilon)
        loss = -self.reduction(y_true * jnp.log(y_preds + 1e-9))
        return loss


class MeanSquaredError(Loss):
    def __init__(self, reduction="auto", name=None):
        super(MeanSquaredError, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        loss = self.reduction(jnp.square(jnp.subtract(y_preds, y_true)))
        return loss


class MeanAbsoluteError(Loss):
    def __init__(self, reduction="auto", name=None):
        super(MeanAbsoluteError, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        loss = self.reduction(jnp.abs(jnp.subtract(y_preds, y_true)))
        return loss


class Huber(Loss):
    def __init__(self, reduction="auto", delta=1.0, name=None):
        super(Huber, self).__init__(reduction, name)
        self.delta = delta

    def call(self, y_true, y_preds):
        loss = optax.huber_loss(y_preds, y_true)
        loss = self.reduction(loss)
        return loss

class BinaryCrossEntropyWithLogits(Loss):
    def __init__(self, reduction="auto", name=None):
        super(BinaryCrossEntropyWithLogits, self).__init__(
            reduction=reduction, name=name
        )

    def call(self, y_true, y_preds):
        return self.reduction(optax.sigmoid_binary_cross_entropy(y_preds, y_true))


class BinaryCrossEntropy(Loss):
    def __new__(cls, *args, **kwargs):
        from_logits = kwargs.pop("with_logits", False)
        if from_logits:
            return BinaryCrossEntropyWithLogits(*args, **kwargs)
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, from_logits=False, reduction="auto", name=None):
        super(BinaryCrossEntropy, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        lhs = y_true * jnp.log(y_preds * self.epsilon)
        rhs = (1 - y_true) * jnp.log(1 - y_preds + self.epsilon)
        loss = -self.reduction(lhs + rhs)
        return loss


class CosineDistance(Loss):
    def __init__(self, reduction="auto", name=None):
        super(CosineDistance, self).__init__(reduction=reduction, name=name)

    def call(self, y_true, y_preds):
        loss = optax.cosine_distance(y_preds, y_true)
        loss = self.reduction(loss)
        return loss


supported_losses = {
    "binary_crossentropy": BinaryCrossEntropy,
    "binary_crossentropy_from_logits": BinaryCrossEntropyWithLogits,
    "categorical_crossentropy": CategoricalCrossEntropy,
    "categorical_crossentropy_from_logits": CategoricalCrossEntropyWithLogits,
    "mse": MeanSquaredError,
    "mean_squared_error": MeanSquaredError,
    "mae": MeanAbsoluteError,
    "mean_absolute_error": MeanAbsoluteError,
    "cosine_distance": CosineDistance,
}


def get(identifier: Union[str, Loss]) -> Loss:
    loss_fn = None
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        loss_fn = supported_losses.get(identifier, None)
    elif isinstance(identifier, Loss) and callable(identifier):
        return identifier

    if loss_fn is None:
        raise ValueError(
            f"Cannot find the specified loss function. Recieved: identifier={identifier}"
        )
    else:
        return loss_fn
