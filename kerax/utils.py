from jax import numpy as jnp  # type: ignore
import jax


def to_categorical(inputs, num_classes, axis=-1):
    return jax.nn.one_hot(inputs, num_classes=num_classes, dtype=jnp.int32, axis=axis)


def to_numbers(inputs, axis=-1):
    return jnp.argmax(inputs, axis=axis)


class Sequence:
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
