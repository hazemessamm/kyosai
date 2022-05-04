import numpy as np
from jax import jit
from kerax import backend


def _check_seed(seed):
    if seed is None:
        return np.random.randint(1e6)
    elif not isinstance(seed, int):
        raise ValueError(
            f"`seed` should be with type int. Recieved: seed={seed} with type={type(seed)}"
        )
    else:
        return seed


def _check_jit(layer):
    if backend.is_jit_enabled():
        layer.call = jit(layer.call)
        layer.call_with_external_weights = jit(layer.call_with_external_weights)
