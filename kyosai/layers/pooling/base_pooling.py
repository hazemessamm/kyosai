from typing import Tuple, Union

from jax import lax
from kyosai.layers.base_layer import Layer


class Pooling(Layer):
    def __init__(
        self,
        pool_size: Union[int, Tuple] = (2, 2),
        strides: Union[int, Tuple] = (2, 2),
        padding: str = "valid",
        dims: int = 2,
        seed: int = None,
        dtype="float32",
        name=None,
        **kwargs,
    ):
        super(Pooling, self).__init__(
            seed=seed, trainable=False, dtype=dtype, name=name
        )
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.dims = dims
        self._validate_init()

        input_shape = kwargs.get("input_shape", False)
        if input_shape:
            self.build(input_shape)

    def _validate_init(self):
        if isinstance(self.pool_size, int):
            self.pool_size = tuple(self.pool_size for _ in range(self.dims))
        elif isinstance(self.pool_size, tuple) and len(self.pool_size) != self.dims:
            self.pool_size += tuple(self.pool_size for _ in range(self.dims))

        if isinstance(self.strides, int):
            self.strides = tuple(self.strides for _ in range(self.dims))
        elif isinstance(self.strides, tuple) and len(self.strides) != self.dims:
            self.strides += tuple(self.strides for _ in range(self.dims))

        self.padding = self.padding.upper()

        self._pool_size = self.pool_size
        self._strides = self.strides

        self.pool_size = (1, *self.pool_size, 1)
        self.strides = (1, *self.strides, 1)

    def compute_output_shape(self, input_shape):
        # lax.reduce_window_shape_tuple() does not accept batch size with None
        # so it's replaced with '1' only in this function
        input_shape = (1, *input_shape[1:])
        padding_vals = lax.padtype_to_pads(
            input_shape, self.pool_size, self.strides, self.padding
        )

        num_dims = tuple(1 for _ in range(self.dims))
        base_dilation = (1, *num_dims, 1)
        window_dilation = (1, *num_dims, 1)

        out_shape = lax.reduce_window_shape_tuple(
            operand_shape=input_shape,
            window_dimensions=self.pool_size,
            window_strides=self.strides,
            padding=padding_vals,
            base_dilation=base_dilation,
            window_dilation=window_dilation,
        )
        return out_shape
