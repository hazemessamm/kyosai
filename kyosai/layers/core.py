import operator as op
from functools import reduce
from typing import Callable, Tuple, Union, List

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.numpy import DeviceArray
from kyosai import activations, backend
from kyosai.layers.base_layer import Layer


class Input(Layer):
    """
    Input Layer that stores the input shape.
    Args:
        shape: the dimension of the input shape.
        dtype: the datatype of the inputs.
        name: name of the `Input` layer.
    """

    def __init__(self, shape: Tuple, dtype: str = "float32", name: str = None):
        super(Input, self).__init__(seed=0, trainable=False, dtype=dtype, name=name)

        self.input_shape = tuple(shape)
        self.built = True

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return (None, *input_shape)

    def call(self, inputs: DeviceArray, **kwargs):
        if inputs.dtype != self.dtype:
            raise ValueError(
                f"`input` should have dtype `{self.dtype}`. Recieved: {inputs.dtype}"
            )
        return inputs

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return inputs


class Dense(Layer):
    """

    Dense Layer, (Layer subclass)

    Args:
        units: Integer, Number of neurons.
        activation: (String, jax.nn.*), stores the activation function.
        kernel_initializer: stores the kernel initializer.
        bias_initializer: stores the bias initializer.
        use_bias: Boolean, whether to enable `bias` vector or not.
        trainable: Boolean, whether to train the weights and the bias (if enabled) or not.
        seed: random seed.
        dtype: datatype of the weights and the bias (if enabled).


    """

    def __init__(
        self,
        units: int,
        activation: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_normal",
        bias_initializer: Union[str, Callable] = "normal",
        use_bias: bool = True,
        trainable: bool = True,
        seed: int = None,
        dtype: str = "float32",
        **kwargs,
    ):
        super(Dense, self).__init__(
            seed=seed, trainable=trainable, dtype=dtype, **kwargs
        )
        self.units = units
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.use_bias = use_bias

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return ((None,) * len(input_shape[1:])) + (self.units,)

    def build(self, input_shape: Tuple):
        k1, k2 = random.split(self.seed)
        self.add_weight(
            key=k1,
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            name=f"{self.name}_kernel",
            trainable=self.trainable,
        )
        if self.use_bias:
            self.add_weight(
                key=k2,
                shape=(self.units,),
                initializer=self.bias_initializer,
                dtype=self.dtype,
                name=f"{self.name}_bias",
                trainable=self.trainable,
            )

        self.input_shape = input_shape
        self.built = True

    def dense_op(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        output = backend.weight_matmul(inputs, weights)
        output = backend.bias_add(output, weights, self.use_bias)
        return backend.apply_activation(output, self.activation)

    def call(self, inputs: DeviceArray, **kwargs):
        "Used during training to pass the parameters while getting the gradients"
        output = self.dense_op(self.weights, inputs)
        return output

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        output = self.dense_op(weights, inputs)
        return output


class Flatten(Layer):
    """
    Flatten Layer, (Layer subclass)
    Args:
        name: String, layer name.

    """

    def __init__(self, name=None, **kwargs):
        super(Flatten, self).__init__(name=name, seed=0, trainable=False, **kwargs)

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return (None, reduce(op.mul, input_shape[1:], 1))

    def flatten_op(self, weights: Tuple, inputs: DeviceArray):
        return jax.numpy.reshape(inputs, (inputs.shape[0], -1))

    def call(self, inputs: DeviceArray, **kwargs):
        "Used during training to pass the parameters while getting the gradients"
        output = self.flatten_op(self.weights, inputs)
        return output

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.flatten_op(weights, inputs)


class Dropout(Layer):
    """
    Dropout Layer, (Layer subclass)
    Args:
        rate: probability of turning of a neuron, accepts probability values between 0 and 1.
        training: stores the mode of the layer, accepts boolean.
        key: Pseudo Random Generator Key, default PRNGKey(1).
    """

    def __init__(self, rate: float, seed: int = None, name: str = None, **kwargs):
        super(Dropout, self).__init__(seed=seed, name=name, **kwargs)
        self.rate = rate

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return input_shape

    def build(self, input_shape: Tuple):
        self.input_shape = input_shape
        self.built = True

    def dropout_op(self, weights: Tuple, inputs: DeviceArray):
        keep = random.bernoulli(self.seed, self.rate, inputs.shape)
        return jnp.where(keep, inputs / self.rate, 0)

    def call(self, inputs: DeviceArray, training=True):
        "Used during training to pass the parameters while getting the gradients"
        return lax.cond(
            training, lambda: self.dropout_op(self.weights, inputs), lambda: inputs
        )

    def call_with_external_weights(
        self, weights: Tuple, inputs: DeviceArray, training=True
    ):
        return lax.cond(
            training, lambda: self.dropout_op(weights, inputs), lambda: inputs
        )


class Activation(Layer):
    """
    Activation Layer, (Layer subclass)
    Args:
        identifier: accepts the activation function as a string or callable.
    """

    def __init__(self, identifier: Union[str, Callable], name=None, **kwargs):
        super(Activation, self).__init__(name=name, seed=0, **kwargs)
        self._identifier = identifier
        self.activation = activations.get(identifier)

    @property
    def identifier(self):
        return self._identifier

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return input_shape

    def activation_op(self, weights: Tuple, inputs: DeviceArray):
        return self.activation(inputs)

    def build(self, input_shape: Tuple):
        self.input_shape = input_shape
        self.built = True

    def call(self, inputs: DeviceArray):
        return self.activation_op(self.weights, inputs)

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray):
        return self.activation_op(weights, inputs)


class Reshape(Layer):
    """
    Reshape Layer, (Layer subclass)
    Args:
        target_shape: accepts the target shape as a `tuple`.
    """

    def __init__(self, target_shape, name=None, **kwargs):
        super(Reshape, self).__init__(name=name, **kwargs)
        self.target_shape = target_shape

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return (None, *self.target_shape)

    def build(self, input_shape: Tuple):
        self.input_shape = input_shape
        self.built = True

    def reshape_op(self, weights, inputs):
        return lax.reshape(inputs, (inputs.shape[0], *self.target_shape))

    def call(self, inputs):
        return self.reshape_op(self.weights, inputs)

    def call_with_external_weights(self, weights, inputs):
        return self.reshape_op(weights, inputs)


class Squeeze(Layer):
    """
    Squeeze Layer, (Layer subclass)
    Args:
        axis: accepts the target axis as a `int`.
    """

    def __init__(self, axis, name=None, **kwargs):
        super(Squeeze, self).__init__(name=name, **kwargs)
        self.axis = axis

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return (*input_shape[: self.axis], *input_shape[self.axis + 1 :])

    def build(self, input_shape: Tuple):
        self.input_shape = input_shape
        self.built = True

    def squeeze_op(self, weights, inputs):
        return lax.squeeze(inputs, (self.axis,))

    def call(self, inputs):
        return self.squeeze_op(self.weights, inputs)

    def call_with_external_weights(self, weights, inputs):
        return self.squeeze_op(weights, inputs)
