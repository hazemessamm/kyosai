from typing import Callable, Tuple, Union, List

from jax import lax
from jax import numpy as jnp
from jax import random
from jax.numpy import DeviceArray
from kyosai.layers.base_layer import Layer
from kyosai import backend


class Conv2D(Layer):
    """
    Convolutional Layer, (Layer Subclass)
    Args:
        filters: Stores number of filters, accepts int
        kernel_size: stores size of each filter, accepts int or tuple of length 2
        strides: stores size of the strides, default (1,1), accepts int or tuple
        padding: padding for the input, accepts "valid" or "same"
        activation: stores the activation function, accepts activation as string or callable
        kernel_initializer: stores the kernel initializer, default "glorot_uniform"
        bias_initializer: stores the bias initializer, default "zeros"
        key: stores Pseudo Random Generator Key, default PRNGKey(1)
        input_dim_order: stores the order of the dimensions, default NHWC
            where N=Batch size, H=Height, W=Width and C=Number of channels
        kernel_dim_order: stores the order of the dimensions, default HWIO
            where H=Height, W=Width, I=Input Size which is the number of channels of the input
            and O=Output size which is the number of the filters
        output_dim_order: stores the order of the dimensions, default NHWC

    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, tuple],
        strides: Union[int, tuple] = (1, 1),
        padding: str = "valid",
        activation: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "normal",
        use_bias: bool = True,
        seed: int = None,
        trainable: bool = True,
        dtype="float32",
        name: str = None,
        **kwargs,
    ):
        super(Conv2D, self).__init__(
            seed=seed, trainable=trainable, dtype=dtype, name=name, **kwargs
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.built = False
        self.use_bias = use_bias

        # (input_dim_order, kernel_dim_order, output_dim_order)
        self.dimension_numbers = ("NHWC", "HWIO", "NHWC")
        self._validate_init()

    def compute_output_shape(self, input_shape):
        if len(input_shape) < 4:
            input_shape = (None, *input_shape)

        return lax.conv_general_shape_tuple(
            lhs_shape=input_shape,
            rhs_shape=self.compute_kernel_shape(input_shape),
            window_strides=self.strides,
            padding=self.padding,
            dimension_numbers=self.dimension_numbers,
        )

    def compute_kernel_shape(self, input_shape: Tuple):
        return (*self.kernel_size, input_shape[-1], self.filters)

    def compute_bias_shape(self):
        return (self.filters,)

    def _validate_init(self):
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        elif isinstance(self.strides, tuple) and len(self.strides) == 1:
            self.strides *= 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        elif isinstance(self.kernel_size, tuple) and len(self.kernel_size) == 1:
            self.kernel_size *= 2

        if self.padding == "causal":
            raise ValueError(
                f"`causal` padding is only allowed in `Conv1D` layer. Recieved padding={self.padding}"
            )

    def build(self, input_shape: Tuple):
        "Initializes the Kernel and stores the Conv2D weights"
        input_shape = (None, *input_shape[-3:])

        k1, k2 = random.split(self.seed)
        kernel_shape = self.compute_kernel_shape(input_shape)
        self.add_weight(
            key=k1,
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            name=f"{self.name}_kernel",
            trainable=self.trainable,
        )
        if self.use_bias:
            bias_shape = self.compute_bias_shape()
            self.add_weight(
                key=k2,
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                name=f"{self.name}_bias",
                trainable=self.trainable,
            )

        self.input_shape = input_shape
        self.dn = lax.conv_dimension_numbers(
            input_shape, kernel_shape, self.dimension_numbers
        )
        self.built = True

    def convolution_op(self, weights: Tuple, inputs: DeviceArray):
        output = backend.apply_conv_general_dilated(
            inputs, weights, self.strides, self.padding, self.dn
        )
        output = backend.bias_add(output, weights, self.use_bias)
        output = backend.apply_activation(output, self.activation)
        return output

    def call(self, inputs: DeviceArray, **kwargs):
        return self.convolution_op(self.weights, inputs)

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.convolution_op(weights, inputs)
