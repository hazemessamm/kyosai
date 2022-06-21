from typing import NamedTuple, Union

import optax  # type:ignore
from jax import jit
from jax import numpy as jnp

from kerax import backend


class Reducer:
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        if reduction == "mean":
            self.reduce = jnp.mean
        elif reduction == "sum":
            self.reduce = jnp.sum
        elif reduction == "none" or reduction is None:
            self.reduce = lambda inputs: inputs
        else:
            raise ValueError(
                f'`reduction` can only be `"mean"`, `"sum"`, '
                f'`"none"` or `None`. Recieved: {reduction}'
            )

    def __call__(self, inputs):
        return self.reduce(inputs)


class Loss:
    def __init__(self, reduction="mean", name=None):
        self.reduction = Reducer(reduction)
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        self.epsilon = 1e-7

    @property
    def __name__(self):
        return self.name

    def call(self, y_true, y_preds):
        raise NotImplementedError("Must be implemented in subclass")

    def __call__(self, y_true, y_preds):
        return self.call(y_true, y_preds)

    def get_config(self):
        return {"reduction": self.reduction, "name": self.name}


class CategoricalCrossEntropyWithLogits(Loss):
    def __init__(self, reduction="mean", name=None):
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

    def __init__(self, from_logits=False, reduction="mean", name=None):
        super(CategoricalCrossEntropy, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        y_preds = jnp.clip(y_preds, self.epsilon, 1.0 - self.epsilon)
        loss = -self.reduction(y_true * jnp.log(y_preds + 1e-9))
        return loss


class MeanSquaredError(Loss):
    def __init__(self, reduction="mean", name=None):
        super(MeanSquaredError, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        loss = self.reduction(jnp.square(jnp.subtract(y_preds, y_true)))
        return loss


class MeanAbsoluteError(Loss):
    def __init__(self, reduction="mean", name=None):
        super(MeanAbsoluteError, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        loss = self.reduction(jnp.abs(jnp.subtract(y_preds, y_true)))
        return loss


class Huber(Loss):
    def __init__(self, reduction="mean", delta=1.0, name=None):
        super(Huber, self).__init__(reduction, name)
        self.delta = delta

    def call(self, y_true, y_preds):
        loss = optax.huber_loss(y_preds, y_true)
        loss = self.reduction(loss)
        return loss


class BinaryCrossEntropyWithLogits(Loss):
    def __init__(self, reduction="mean", name=None):
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

    def __init__(self, from_logits=False, reduction="mean", name=None):
        super(BinaryCrossEntropy, self).__init__(reduction, name)

    def call(self, y_true, y_preds):
        y_pred = jnp.clip(y_preds, self.epsilon, 1 - self.epsilon)
        return -self.reduction(
            (y_true * jnp.log(y_pred + self.epsilon))
            + ((1 - y_true) * jnp.log(1 - y_pred + self.epsilon))
        )


class CosineDistance(Loss):
    def __init__(self, reduction="mean", name=None):
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
