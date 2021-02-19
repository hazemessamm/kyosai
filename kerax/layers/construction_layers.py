from jax import nn
import set_path
from activations import activations

#Construction layer for the activation
def Activation(identifier):
    activation = activations.get(identifier)

    def init_fn(input_shape):
        return input_shape, ()
    
    def apply_fn(x, params):
        out = activation(x)
        return out
    
    return init_fn, apply_fn

