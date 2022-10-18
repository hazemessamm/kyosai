from typing import Callable, Tuple, Union

from jax import lax
from jax import random
from kyosai import backend
from kyosai.layers.base_layer import Layer


class Conv1DTranspose(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, tuple],
        strides: Union[int, tuple] = (1,),
        padding: str = "valid",
        activation: Union[str, Callable] = None,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        bias_initializer: Union[str, Callable] = "normal",
        use_bias: bool = True,
        seed: int = None,
        trainable: bool = True,
        dtype: str = "float32",
        name: str = None,
        **kwargs,
    ):
        super(Conv1DTranspose, self).__init__(
            seed=seed, trainable=trainable, dtype=dtype, name=name
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.activation = self.get_activation(activation)
        self.kernel_initializer = self.get_initializer(kernel_initializer)
        self.bias_initializer = self.get_initializer(bias_initializer)
        self.built = False
        self.use_bias = use_bias
        # (input_dim_order, kernel_dim_order, output_dim_order)
        self.dimension_numbers = ("NHC", "HIO", "NHC")
        self._validate_init()
        shape = kwargs.get("shape", False) or kwargs.pop("input_shape", False)
        if shape:
            self.build(shape)

    def compute_output_shape(self, input_shape):
        kernel_shape = self.compute_kernel_shape(self.input_shape)
        return lax.conv_transpose_shape_tuple(
            input_shape,
            kernel_shape,
            self.strides,
            self.padding,
            self.dimension_numbers,
        )

    def compute_kernel_shape(self, input_shape: Tuple):
        return (*self.kernel_size, input_shape[-1], self.filters)

    def compute_bias_shape(self):
        return (self.filters,)

    def _validate_init(self):
        if isinstance(self.strides, int):
            self.strides = (self.strides,)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size,)

        # TODO: remove it and implement causal padding.
        if self.padding == "causal":
            raise ValueError(
                f"`causal` padding is not implemented yet in `Conv1D` layer. Recieved padding={self.padding}"
            )

    def build(self, input_shape):
        if len(input_shape) > 3:
            raise ValueError(
                f"`input_shape` should have only 3 dimensions (H, C). Recieved: input_shape={input_shape}"
            )

        input_shape = (None, *input_shape[-2:])
        self.input_shape = input_shape
        k1, k2 = random.split(self.seed)
        kernel_shape = self.compute_kernel_shape(input_shape)
        self.kernel_weights = self.add_weight(
            key=k1,
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            name=f"{self.name}_kernel",
            trainable=self.trainable,
        )
        if self.use_bias:
            bias_shape = self.compute_bias_shape()
            self.bias_weights = self.add_weight(
                key=k2,
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                name=f"{self.name}_bias",
                trainable=self.trainable,
            )

        self.dn = lax.conv_dimension_numbers(
            input_shape, kernel_shape, self.dimension_numbers
        )
        self.built = True

    def convolution_transpose_op(self, weights, inputs, **kwargs):
        output = lax.conv_transpose(
            inputs, weights[0], self.strides, self.padding, dimension_numbers=self.dn
        )

        output = backend.bias_add(output, weights, self.use_bias)
        output = backend.apply_activation(output, self.activation)
        return output

    def call(self, inputs, **kwargs):
        return self.convolution_transpose_op(self.weights, inputs)

    def call_with_external_weights(self, weights, inputs, **kwargs):
        return self.convolution_transpose_op(weights, inputs)
