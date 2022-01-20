import jax
from jax import random
from jax import lax
from jax import numpy as jnp




def _pooling_layer(reducer, init_val, rescaler=None):
  def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
    """Layer construction function for a pooling layer."""
    strides = strides or (1,) * len(window_shape)
    rescale = rescaler(window_shape, strides, padding) if rescaler else None

    if spec is None:
      non_spatial_axes = 0, len(window_shape) + 1
    else:
      non_spatial_axes = spec.index('N'), spec.index('C')

    for i in sorted(non_spatial_axes):
      window_shape = window_shape[:i] + (1,) + window_shape[i:]
      strides = strides[:i] + (1,) + strides[i:]

    def init_fun(rng, input_shape):
        print(input_shape, window_shape, strides, padding)
        padding_vals = lax.padtype_to_pads(input_shape, window_shape,
                                         strides, padding)
        ones = (1,) * len(window_shape)
        out_shape = lax.reduce_window_shape_tuple(
            input_shape, window_shape, strides, padding_vals, ones, ones)
        return out_shape, ()
    def apply_fun(params, inputs, **kwargs):
      out = lax.reduce_window(inputs, init_val, reducer, window_shape,
                              strides, padding)
      return rescale(out, inputs, spec) if rescale else out
    return init_fun, apply_fun
  return PoolingLayer

MaxPool = _pooling_layer(lax.max, -jnp.inf)

init, apply = MaxPool((2,2), strides=(2,2))


out, _ = init(random.PRNGKey(1), (1, 28, 28, 3))

print(out)