from jax import numpy as jnp #type: ignore


def to_categorical(inputs, num_classes=None):
    if num_classes is None:
        num_classes = jnp.max(inputs, axis=-1)
    return (inputs[:, None] == jnp.arange(num_classes)).astype(jnp.int32)


def to_numbers(inputs, axis=-1):
  return jnp.argmax(inputs, axis=axis)


class Sequence:
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
