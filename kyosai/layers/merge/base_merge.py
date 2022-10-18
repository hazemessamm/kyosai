from typing import Tuple
from kyosai.layers.base_layer import Layer


class Merge(Layer):
    """
    Merge Layer, (Layer subclass)
    Args:
        seed: random seed.
        name: name of the `Merge` layer.
    """

    def __init__(self, seed: int = None, name: str = None, **kwargs):
        super(Merge, self).__init__(seed=seed, name=name)
        self.supports_specific_axis = False
        self.supported_axis = None

    def _check_shapes(self, input_shapes):
        shapes = set(input_shapes)
        if len(shapes) != 1:
            raise ValueError(
                f"`input_shape` contains unmatched shapes. Recieved: shapes={shapes}"
            )

    def _check_axis(self, input_shapes):
        if self.supports_specific_axis:
            axis_list = set()
            for _input_shape in input_shapes:
                axis_list.add(_input_shape[self.supported_axis])

            if len(axis_list) != 1:
                raise ValueError(
                    f"`input_shape` contains unmatched axis."
                    f"Recieved: axis={self.supported_axis} and different axis are {axis_list}"
                )

    def build(self, input_shape: Tuple):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError(
                f"`input_shape` should be a list of input shapes. Recieved: input_shape={input_shape}"
            )
        elif len(input_shape) <= 1:
            raise ValueError(
                f"`input_shape` should be a list with length bigger than 1. Recieved: input_shape={input_shape}"
            )
        else:
            self._check_shapes(input_shape)
            self._check_axis(input_shape)

        self.input_shape = input_shape
        self.built = True
