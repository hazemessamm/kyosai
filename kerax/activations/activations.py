from jax import nn #type: ignore



acts = {
    'celu': nn.celu,
    'elu': nn.elu,
    'gelu': nn.gelu,
    'glu': nn.glu,
    'hard_sigmoid': nn.hard_sigmoid,
    'hard_silu': nn.hard_silu,
    'hard_swish': nn.hard_swish,
    'hard_tanh': nn.hard_tanh,
    'log_softmax': nn.log_softmax,
    'logsumexp': nn.logsumexp,
    'relu': nn.relu,
    'relu6': nn.relu6,
    'selu': nn.selu,
    'sigmoid': nn.sigmoid,
    'silu': nn.silu,
    'soft_sign': nn.soft_sign,
    'softmax': nn.softmax,
    'softplus': nn.softplus,
    'swish': nn.swish
}

def get(identifier):
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        return acts.get(identifier, None)
    else:
        raise Exception("Cannot find the specified identifier")


def LeakyReLU(x, negative_slope=0.01):
    return nn.LeakyReLU(x, negative_slope=negative_slope)