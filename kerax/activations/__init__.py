from jax.nn import (celu, elu, gelu, glu, hard_sigmoid, hard_silu, hard_swish,
                    hard_tanh, leaky_relu, log_softmax, logsumexp, relu, relu6,
                    selu, sigmoid, silu, soft_sign, softmax, softplus, swish)

ReLU = relu
leaky_relu = leaky_relu
Softmax = softmax
SoftPlus = softplus
Swish = swish
ReLU6 = relu6

supported_activations = {
    'celu': celu,
    'elu': elu,
    'gelu': gelu,
    'glu': glu,
    'hard_sigmoid': hard_sigmoid,
    'hard_silu': hard_silu,
    'hard_swish': hard_swish,
    'hard_tanh': hard_tanh,
    'log_softmax': log_softmax,
    'logsumexp': logsumexp,
    'relu': relu,
    'relu6': relu6,
    'selu': selu,
    'sigmoid': sigmoid,
    'silu': silu,
    'soft_sign': soft_sign,
    'softmax': softmax,
    'softplus': softplus,
    'swish': swish,
    'leaky_relu': leaky_relu,
}

def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        return supported_activations.get(identifier, None)
    else:
        raise Exception("Cannot find the specified identifier")
