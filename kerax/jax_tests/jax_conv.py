# import jax
# from jax import random
# from jax import lax
# from jax import numpy as jnp
# import operator as op
# import functools



# def Flatten():
#   """Layer construction function for flattening all but the leading dim."""
#   def init_fun(rng, input_shape):
#     output_shape = input_shape[0], functools.reduce(op.mul, input_shape[1:], 1)
#     return output_shape, ()
#   def apply_fun(params, inputs, **kwargs):
#     return jnp.reshape(inputs, (inputs.shape[0], -1))
#   return init_fun, apply_fun
# Flatten = Flatten()