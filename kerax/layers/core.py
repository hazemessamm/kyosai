import operator as op
from functools import reduce
from typing import Callable, Tuple, Union

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.numpy import DeviceArray
from kerax import activations
from kerax.layers.base_layer import Layer


class Input(Layer):
    """
    Input Layer that stores the input shape
    """

    def __init__(self, shape: Tuple = None, dtype: str = "float32", name: str = None):
        super(Input, self).__init__(seed=0, trainable=False, dtype=dtype, name=name)
        if not shape or not isinstance(shape, tuple):
            raise Exception(
                f"shape should have value in a tuple, found {shape} with type {type(shape)}"
            )
        if shape:
            shape = tuple(shape)
            self._shape, self._input_shape = shape, shape
            self.built = True
        else:
            raise Exception(
                f"Error in {self.name}, input shape must be provided for the Input Layer"
            )

    @property
    def shape(self):
        return (None, *self._shape)

    @property
    def input_shape(self):
        return self._input_shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape

    def call(self, inputs: DeviceArray, **kwargs):
        if inputs.dtype != self.dtype:
            raise ValueError(
                f"`input` should have dtype `{self.dtype}`. Recieved: {inputs.dtype}"
            )
        return inputs

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray, **kwargs):
        return inputs


class Dense(Layer):
    """

    Dense Layer, (Layer subclass)

    params:
    units: stores number of columns (neurons)
    activation: stores the activation function, default None
    kernel_initializer: stores the kernel initializer, default "glorot_normal"
    bias_initializer: stores the bias initializer, default "normal"

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

    @property
    def shape(self):
        return (None, self.units)

    def compute_kernel_shape(self):
        return self.kernel_weights.shape

    def compute_bias_shape(self):
        return self.bias_weights.shape

    def compute_output_shape(self):
        return (None, self.units)

    def build(self, input_shape: Tuple):
        self._input_shape = (input_shape[-1],)
        k1, k2 = random.split(self.seed)
        self.kernel_weights = self.add_weight(
            k1,
            (input_shape[-1], self.units),
            self.kernel_initializer,
            self.dtype,
            f"{self.name}_kernel",
            trainable=self.trainable,
        )
        if self.use_bias:
            self.bias_weights = self.add_weight(
                k2,
                (self.units,),
                self.bias_initializer,
                self.dtype,
                f"{self.name}_bias",
                trainable=self.trainable,
            )
        self.built = True

    def dense_op(self, params: Tuple, inputs: DeviceArray, **kwargs):
        output = jnp.matmul(inputs, params[0])
        return jnp.add(output, params[1]) if self.use_bias else output

    def call(self, inputs: DeviceArray, **kwargs):
        "Used during training to pass the parameters while getting the gradients"
        output = self.dense_op(self.params, inputs)

        if self.activation:
            output = self.activation(output)
        return output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray, **kwargs):
        output = self.dense_op(params, inputs)

        if self.activation:
            output = self.activation(output)
        return output


class Flatten(Layer):
    """
    Flatten Layer, (Layer subclass)
    params:
    key: Pseudo Random Generator Key, default PRNGKey(1)

    """

    def __init__(self, **kwargs):
        super(Flatten, self).__init__(seed=0, trainable=False, **kwargs)

    @property
    def compute_output_shape(self):
        return (self.input_shape[0], reduce(op.mul, self.input_shape[1:], 1))

    @property
    def shape(self):
        return (None, reduce(op.mul, self.input_shape[1:], 1))

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def flatten_op(self, params: Tuple, inputs: DeviceArray):
        return lax.reshape(inputs, (inputs.shape[0], *self.shape[1:]))

    def call(self, inputs: DeviceArray, **kwargs):
        "Used during training to pass the parameters while getting the gradients"
        output = self.flatten_op(self.params, inputs)
        return output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray, **kwargs):
        return self.flatten_op(params, inputs)


class Dropout(Layer):
    """
    Dropout Layer, (Layer subclass)
    params:
    rate: probability of turning of a neuron, accepts probability values between 0 and 1
    training: stores the mode of the layer, accepts boolean
    key: Pseudo Random Generator Key, default PRNGKey(1)
    """

    def __init__(self, rate: float, seed: int = None, name: str = None, **kwargs):
        super(Dropout, self).__init__(seed=seed, name=name, **kwargs)
        self.rate = rate

    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._input_shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def dropout_op(self, params: Tuple, inputs: DeviceArray):
        keep = random.bernoulli(self.seed, self.rate, inputs.shape)
        return jnp.where(keep, inputs / self.rate, 0)

    def identity_op(self, params, inputs):
        return inputs

    def call(self, inputs: DeviceArray, training=True):
        "Used during training to pass the parameters while getting the gradients"
        return lax.cond(training, lambda: self.dropout_op(self.params, inputs), lambda: inputs)

    def call_with_external_weights(
        self, params: Tuple, inputs: DeviceArray, training=True
    ):
        return lax.cond(training, lambda: self.dropout_op(self.params, inputs), lambda: inputs)


class Activation(Layer):
    """
    Activation Layer, (Layer subclass)
    params:
    identifier: accepts the activation function as a string or callable
    """

    def __init__(self, identifier: Union[str, Callable], **kwargs):
        super(Activation, self).__init__(seed=0, **kwargs)
        self._identifier = identifier
        self.activation = activations.get(identifier)

    @property
    def identifier(self):
        return self._identifier

    @property
    def compute_output_shape(self):
        return self._input_shape

    @property
    def shape(self):
        return self._input_shape

    def activation_op(self, params: Tuple, inputs: DeviceArray):
        return self.activation(inputs)

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def call(self, inputs: DeviceArray):
        return self.activation_op(self.params, inputs)

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        return self.activation_op(params, inputs)


class Reshape(Layer):
    def __init__(self, target_shape, name=None):
        super(Reshape, self).__init__(name=name)
        self.target_shape = target_shape

    @property
    def shape(self):
        return (None, *self.target_shape)

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self.built = True

    def reshape_op(self, params, inputs):
        return lax.reshape(inputs, (inputs.shape[0], *self.target_shape))

    def call(self, inputs):
        return self.reshape_op(self.params, inputs)

    def call_with_external_weights(self, params, inputs):
        return self.reshape_op(params, inputs)


class Squeeze(Layer):
    def __init__(self, axis, name=None):
        super(Squeeze, self).__init__(name=name)
        self.axis = axis

    @property
    def shape(self):
        return self._shape

    def build(self, input_shape: Tuple):
        self._input_shape = input_shape
        self._shape = (*input_shape[: self.axis], *input_shape[self.axis + 1:])
        self.built = True

    def squeeze_op(self, params, inputs):
        return lax.squeeze(inputs, (self.axis,))

    def call(self, inputs):
        return self.squeeze_op(self.params, inputs)

    def call_with_external_weights(self, params, inputs):
        return self.squeeze_op(params, inputs)
