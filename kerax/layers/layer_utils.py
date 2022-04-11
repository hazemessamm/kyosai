from kerax import backend
from jax import jit


def _check_seed(seed):
    if seed is None:
        raise ValueError(f"`seed` should be with type int. Recieved: seed={seed}")


def _check_jit(layer):
    if backend.is_jit_enabled():
        layer.call = jit(layer.call)
