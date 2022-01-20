# from jax import nn #type: ignore
    
# def celu(x):
#     return nn.celu(x)

# def elu(x):
#     return nn.elu(x)

# def gelu(x):
#     return nn.gelu(x)

# def glu(x):
#     return nn.glu(x)

# def hard_sigmoid(x):
#     return nn.hard_sigmoid(x)

# def hard_silu(x):
#     return nn.hard_silu(x)

# def hard_swish(x):
#     return nn.hard_swish(x)

# def hard_tanh(x):
#     return nn.hard_tanh(x)

# def log_softmax(x):
#     return nn.log_softmax(x)

# def logsumexp(x):
#     return nn.logsumexp(x)

# def relu(x):
#     return nn.relu(x)

# def relu6(x):
#     return nn.relu6(x)

# def selu(x):
#     return nn.selu(x)

# def sigmoid(x):
#     return nn.sigmoid(x)

# def silu(x):
#     return nn.silu(x)

# def soft_sign(x):
#     return nn.soft_sign(x)

# def softmax(x):
#     return nn.softmax(x)

# def softplus(x):
#     return nn.softplus(x)

# def swish(x):
#     return nn.swish(x)

# def LeakyReLU(x, negative_slope=0.01):
#     return nn.LeakyReLU(x, negative_slope=negative_slope)

# ReLU = relu
# leaky_relu = LeakyReLU
# Softmax = softmax
# SoftPlus = softplus
# Swish = swish
# ReLU6 = relu6


# supported_activations = {
#     'celu': celu,
#     'elu': elu,
#     'gelu': gelu,
#     'glu': glu,
#     'hard_sigmoid': hard_sigmoid,
#     'hard_silu': hard_silu,
#     'hard_swish': hard_swish,
#     'hard_tanh': hard_tanh,
#     'log_softmax': log_softmax,
#     'logsumexp': logsumexp,
#     'relu': relu,
#     'relu6': relu6,
#     'selu': selu,
#     'sigmoid': sigmoid,
#     'silu': silu,
#     'soft_sign': soft_sign,
#     'softmax': softmax,
#     'softplus': softplus,
#     'swish': swish
# }

# def get(identifier):
#     if identifier is None:
#         return None
#     elif callable(identifier):
#         return identifier
#     elif isinstance(identifier, str):
#         return supported_activations.get(identifier, None)
#     else:
#         raise Exception("Cannot find the specified identifier")
