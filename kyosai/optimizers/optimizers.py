import optax
from jax import jit


class Optimizer:
    """
    Optimizer base class
    All optimizers should be a subclass from this class
    and all optimizers should follow JAX optimizers rules

    Args:
        loss_fn: stores the loss function to get the gradients of the loss function with respect to the params
        model: stores the model to update it's weights every step
        learning_rate: stores the learning rate (step_size), default: 0.0001
    """

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.step_index = 0
        self._initialized = False
        self._apply_updates = jit(optax.apply_updates)

    def _initialize(self, weights):
        self._optimizer_state = self._optimizer.init(weights)
        self._initialized = True

    def minimize(self, weights, grads):
        "Updates the model weights"
        if not self._initialized:
            self._initialize(weights)
        # returns new optimizer state by calling the update function
        updates, self._optimizer_state = self._optimizer_update(
            grads, self._optimizer_state
        )
        # Apply new weights on the current weights
        weights = self._apply_updates(weights, updates)
        return weights


class SGD(Optimizer):
    """
    Optimizer subclass

    Args:
        learning_rate: stores the learning rate (step_size), default: 0.0001
    """

    def __init__(self, learning_rate=0.001, momentum=None, nesterov=False):
        super(SGD, self).__init__(learning_rate=learning_rate)
        """
        Initializes the Stochastic Gradient Descent
        returns initializer function, update function and get params function
        init function just takes the current model weights
        update function takes the step index, gradients and the current optimizer state
        get params takes the optimizer state and returns the weights
        """
        self._momentum = momentum
        self._nesterov = nesterov
        self._optimizer = optax.sgd(
            learning_rate=learning_rate, momentum=momentum, nesterov=nesterov
        )

        self._optimizer_update = jit(self._optimizer.update)


class Adam(Optimizer):
    """
    Optimizer subclass

    Args:
        learning_rate: stores the learning rate (step_size), default: 0.0001
        beta_1: a positive scalar value for beta_1, the exponential decay rate for the first moment estimates
        beta_2: a positive scalar value for beta_2, the exponential decay rate for the second moment estimates
        epsilon: a positive scalar value for epsilon, a small constant for numerical stability
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        super(Adam, self).__init__(learning_rate=learning_rate)
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._optimizer = optax.adam(
            learning_rate=learning_rate, b1=beta_1, b2=beta_2, eps=epsilon
        )
        self._optimizer_update = jit(self._optimizer.update)


class Adagrad(Optimizer):
    """
    Optimizer subclass

    Args:
        learning_rate: stores the learning rate (step_size), default: 0.0001
        momentum: a positive scalar value for momentum
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        initial_accumulator_value: float = 0.1,
        eps: float = 1e-7,
    ):
        super(Adagrad, self).__init__(learning_rate=learning_rate)
        self._initial_accumulator_value = initial_accumulator_value
        self._eps = eps
        self._optimizer = optax.adagrad(
            learning_rate=learning_rate,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        )
        self._optimizer_update = jit(self._optimizer.update)


class RMSProp(Optimizer):
    """
    Optimizer subclass

    Args:
        loss_fn: stores the loss function to get the gradients of the loss function with respect to the weights
        model: stores the model to update it's weights every step
        learning_rate: stores the learning rate (step_size)
        momentum: a positive scalar value for momentum
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-8,
        initial_scale: float = 0,
        centered: bool = False,
        momentum: float or None = None,
        nesterov: bool = False,
    ):
        super(RMSProp, self).__init__(learning_rate=learning_rate)
        self._decay = decay
        self._eps = eps
        self._initial_scale = initial_scale
        self._centered = centered
        self._momentum = momentum
        self._nesterov = nesterov
        self._optimizer = optax.rmsprop(
            learning_rate=learning_rate,
            decay=decay,
            eps=eps,
            initial_scale=initial_scale,
            centered=centered,
            momentum=momentum,
            nesterov=nesterov,
        )
        self._optimizer_update = jit(self._optimizer.update)


supported_optimizers = {
    "sgd": SGD,
    "adam": Adam,
    "adagrad": Adagrad,
    "rmsprop": RMSProp,
}


def get(identifier):
    optimizer = None
    if identifier is None:
        return None
    elif isinstance(identifier, str):
        optimizer = supported_optimizers.get(identifier, None)
    elif isinstance(identifier, Optimizer):
        return identifier
    if optimizer is None:
        raise Exception("Cannot find the specified optimizer")
    return optimizer
