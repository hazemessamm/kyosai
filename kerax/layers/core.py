import inspect
import operator as op
from functools import reduce
from typing import Any, Callable, List, Tuple, Union

from jax import lax
from jax import numpy as jnp
from jax import random
from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kerax import activations, backend
from kerax.engine.containers import NodeContainer, Weight
from kerax.initializers import Initializer, initializers
from kerax.layers import layer_utils
from numpy import ndarray
from jax.interpreters.partial_eval import DynamicJaxprTracer


# Not used or ready yet
class InputSpec:
    def __init__(self):
        self.input_shape = None
        self.valid_ndims = None
        self._internal_input_shape = None

    def build(self, input_shape, valid_ndims):
        if valid_ndims != len(input_shape):
            raise Exception(
                f"number of dims in input_shape does not match the required number of dims, found {len(input_shape)} expected {valid_ndims}"
            )

        self.input_shape = input_shape
        self.valid_ndims = valid_ndims
        self._internal_input_shape = (None, *input_shape)


class Layer:
    def __init__(
        self,
        seed: int = None,
        trainable: bool = True,
        dtype: str = "float32",
        name: str = None,
        **kwargs,
    ):
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        layer_utils._check_jit(self)

        # Stores the layer params
        self._params = []

        # Stores the previous layer
        self._node_container = NodeContainer()
        self.seed = PRNGKey(layer_utils._check_seed(seed))
        self.trainable = trainable
        self.dtype = dtype or backend.precision()
        self.built = False
        self._has_nested_layers = False
        self._is_nested = False
        self._validated = False
        self._required_num_inputs = None
        self._depth = 0
        self._validate_layer_options()

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Layer):
            if not self._has_nested_layers:
                self._has_nested_layers = True
                self._nested_layers = [__value]
            else:
                self._nested_layers.append(__value)
            __value._is_nested = True
        return super().__setattr__(__name, __value)

    def _validate_layer_options(self):
        call_params = inspect.signature(self.call).parameters
        self._required_number_inputs = len(call_params)

        call_with_external_weights_params = inspect.signature(
            self.call_with_external_weights
        ).parameters
        if "params" not in call_with_external_weights_params:
            raise ValueError(
                f"`params` argument should be added as the first argument in `call_with_external_weights` function. Recieved: {call_with_external_weights_params.keys()}"
            )

    @property
    def shape(self):
        return None

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        raise NotImplementedError("should be implemented in a subclass.")

    @property
    def input_shape(self):
        if self.built:
            return self._input_shape
        raise Exception(f"Error in {self.name}, Layer is not built yet")

    def compute_output_shape(self, input_shape):
        raise NotImplementedError("Should be implemented in a subclass")

    @property
    def weights(self):
        "returns weights"
        return self._params

    @property
    def params(self):
        params = [param.get_weights() for param in self._params]
        if self._has_nested_layers:
            nested_params = tuple(layer.params for layer in self._nested_layers)
            params.extend(nested_params)
        return tuple(params)

    @property
    def named_params(self):
        return {param.name: param.get_weights() for param in self._params}

    def build(self, input_shape: Tuple):
        return NotImplementedError("Should be implemented in a subclass")

    def get_initializer(self, identifier: Union[str, Initializer]):
        "Returns the specified initializer"
        return initializers.get(identifier)

    def get_activation(self, identifier: Union[str, Initializer]):
        "Returns the specified activation"
        return activations.get(identifier)

    def connect(self, layer):
        "Connects the current layer with the previous layer"
        self._node_container.connect_nodes(self, layer)

    def add_weight(
        self,
        key: PRNGKey,
        shape: Tuple,
        initializer: Initializer,
        dtype: str,
        name: str,
        trainable: bool,
    ):
        weight = Weight(key, shape, initializer, dtype, name, trainable)
        self._params.append(weight)
        return weight

    def get_weights(self):
        return self._params

    def set_weights(self, new_weights: Tuple):
        if len(new_weights) > 0:
            if self._has_nested_layers:
                for layer in self._nested_layers:
                    for w1, w2 in zip(layer._params, new_weights):
                        w1.set_weights(w2)
            else:
                for w1, w2 in zip(self._params, new_weights):
                    w1.set_weights(w2)

    def update_weights(self, new_weights: Tuple):
        if len(new_weights) > 0:
            for w_old, w_new in zip(self._params, new_weights):
                w_old.update_weights(w_new)

            if self._has_nested_layers:
                for layer, new_weight in zip(self._nested_layers, new_weights):
                    for w_old, w_new in zip(layer._params, new_weight):
                        w_old.update_weights(w_new)

    def check_shape_if_built(self, layers):
        if isinstance(layers, (list, tuple)):
            for l in layers:
                if self.input_shape != l.shape:
                    raise ValueError(
                        f"This input shape of layer {self.name} does not match the output shape of layer {l.name}. Expected: {self.input_shape}. Recieved: {l.shape}"
                    )
        else:
            if self.input_shape != layers.shape:
                raise ValueError(
                    f"This input shape of layer {self.name} does not match the output shape of layer {layers.name}. Expected: {self.input_shape}. Recieved: {layers.shape}"
                )

    def __call__(self, *inputs, **kwargs):
        if not self._required_num_inputs:
            if len(inputs) > 1:
                inputs = inputs[0]
                args = inputs[1:]
            else:
                inputs = inputs[0]
                args = ()
            has_multiple_inputs = False
        else:
            inputs = inputs[: self._required_num_inputs]
            args = inputs[self._required_num_inputs :]
            has_multiple_inputs = True

        if not self.built:
            if isinstance(inputs, Layer):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            elif isinstance(inputs, (ndarray, DeviceArray, DynamicJaxprTracer)):
                self.build(inputs.shape)
                if has_multiple_inputs:
                    return self.call(*inputs, *args, **kwargs)
                else:
                    return self.call(inputs, *args, **kwargs)
            elif isinstance(inputs, (list, tuple)):
                self.build([_input.shape for _input in inputs])
                if all([isinstance(i, Layer) for i in inputs]):
                    self.connect(inputs)
                    return self
                else:
                    if has_multiple_inputs:
                        return self.call(*inputs, *args, **kwargs)
                    else:
                        return self.call(inputs, *args, **kwargs)
            else:
                raise ValueError(
                    f"`inputs` should be with type `Layer`, `ndarray`, `DeviceArray`, `list` or `tuple`. Recieved: {type(inputs)}"
                )
        else:
            if isinstance(inputs, Layer):
                self.check_shape_if_built(inputs)
                self.connect(inputs)
                return self
            else:
                if has_multiple_inputs:
                    return self.call(*inputs, *args, **kwargs)
                else:
                    return self.call(inputs, *args, **kwargs)

    def call(self, inputs: DeviceArray):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    def __repr__(self):
        if self.built:
            return f"<{self.name} Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return f"<{self.name} Layer>"

    def __name__(self):
        return self.name


class Input(Layer):
    """
    Input Layer that stores the input shape
    params:
    shape: takes a tuple (0, H, W, C)
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

    def call(self, inputs: DeviceArray):
        return inputs

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
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

    def dense_op(self, params: Tuple, inputs: DeviceArray):
        output = jnp.matmul(inputs, params[0])
        return jnp.add(output, params[1]) if self.use_bias else output

    def call(self, inputs: DeviceArray):
        "Used during training to pass the parameters while getting the gradients"
        output = self.dense_op(self.params, inputs)
        return lax.cond(
            self.activation != None, lambda: self.activation(output), lambda: output
        )

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        output = self.dense_op(params, inputs)
        return lax.cond(
            self.activation != None, lambda: self.activation(output), lambda: output
        )


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

    def call(self, inputs: DeviceArray):
        "Used during training to pass the parameters while getting the gradients"
        output = self.flatten_op(self.params, inputs)
        return output

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        return self.flatten_op(params, inputs)


class Dropout(Layer):
    """
    Dropout Layer, (Layer subclass)
    params:
    rate: probability of turning of a neuron, accepts probability values between 0 and 1
    training: stores the mode of the layer, accepts boolean
    key: Pseudo Random Generator Key, default PRNGKey(1)
    """

    def __init__(self, rate: float, seed: int = None, **kwargs):
        super(Dropout, self).__init__(seed=seed, **kwargs)
        self.rate = rate

    @property
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

    def call(self, inputs: DeviceArray, training=False):
        "Used during training to pass the parameters while getting the gradients"
        return lax.cond(
            training, lambda: self.dropout_op(self.params, inputs), lambda: inputs
        )

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray):
        return self.dropout_op(params, inputs)


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
        self._shape = (*input_shape[: self.axis], *input_shape[self.axis + 1 :])
        self.built = True

    def squeeze_op(self, params, inputs):
        return lax.squeeze(inputs, (self.axis,))

    def call(self, inputs):
        return self.squeeze_op(self.params, inputs)

    def call_with_external_weights(self, params, inputs):
        return self.squeeze_op(params, inputs)
