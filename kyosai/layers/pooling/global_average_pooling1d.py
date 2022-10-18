from kyosai.layers.base_layer import Layer
import jax
from typing import Union, List, Tuple
from jax.numpy import DeviceArray


class GlobalAvgPooling1D(Layer):
    """
    GlobalAvgPooling1D Layer, (Layer subclass)
    Args:
        None
    """

    def __init__(self, name=None, **kwargs):
        super(GlobalAvgPooling1D, self).__init__(
            name=name, seed=0, trainable=False, **kwargs
        )

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        return (None, input_shape[-1])

    def global_avg_pooling_op(self, weights: Tuple, inputs: DeviceArray):
        return jax.numpy.mean(inputs, axis=1)

    def call(self, inputs: DeviceArray, **kwargs):
        "Used during training to pass the parameters while getting the gradients"
        output = self.global_avg_pooling_op(self.weights, inputs)
        return output

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        return self.global_avg_pooling_op(weights, inputs)
