from jax import lax
from jax import numpy as jnp  # type:ignore
from kerax.activations import activations


#Construction layer for the activation
def Activation(identifier):
    activation = activations.get(identifier)

    def init_fn(input_shape):
        return input_shape, ()
    
    def apply_fn(params, inputs):
        return activation(inputs)
    
    return init_fn, apply_fn


def Concatenate(axis):
    def init_fun(input_shape, key):
        return input_shape, ()
    
    def apply_fun(params, inputs):
        return jnp.concatenate(inputs, axis=axis)


def Add():
    def init_fn(input_shape, key):
        return input_shape, ()
    
    def apply_fn(params, x1, x2):
        return lax.add(x1, x2)

    return init_fn, apply_fn


def LSTM():
    #TODO: Implement LSTM Conststuction for LSTM layer
    def init_fun(input_shape):
        pass
    
    def apply_fun(x, params):
        pass

    return init_fun, apply_fun


def GRU():
    #TODO: Implement GRU Conststuction for GRU layer
    def init_fun(input_shape):
        pass
    
    def apply_fun(x, params):
        pass

    return init_fun, apply_fun


def Embedding():
    #TODO: Implement Embedding Conststuction for Embedding layer
    def init_fun(input_shape):
        pass
    
    def apply_fun(x, params):
        pass

    return init_fun, apply_fun


def Attention():
    #TODO: Implement Attention Conststuction for Attention layer
    def init_fun(input_shape):
        pass
    
    def apply_fun(x, params):
        pass

    return init_fun, apply_fun


def MultiHeadAttention():
    #TODO: Implement MultiHeadAttention Conststuction for MultiHeadAttention layer
    def init_fun(input_shape):
        pass
    
    def apply_fun(x, params):
        pass

    return init_fun, apply_fun
