from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kyosai.initializers import Initializer, initializers
import jax


class Weight:
    def __init__(
        self,
        key: PRNGKey,
        shape: tuple,
        initializer: Initializer,
        dtype: str,
        name: str,
        trainable: bool,
    ):
        self.key = key
        self.shape = shape
        self.initializer = initializers.get(initializer)
        self.name = name
        self.trainable = trainable
        self.dtype = dtype
        self.built = False
        self.weights = None
        self._called_from_eval_mode = False

    def get_weights(self):
        if self.weights is None:
            weights = self.initializer(self.key, self.shape, self.dtype)
            self.weights = weights
            return weights
        return self.weights

    def set_weights(self, weights: DeviceArray):
        if self.built:
            if (self.weights.shape == weights.shape) and (self.dtype == weights.dtype):
                self.weights = weights
            else:
                raise ValueError(
                    f"New weights shape does not match"
                    f"the current weights shape, {self.weights.shape} != {weights.shape}"
                )
        else:
            raise ValueError(
                "Weights are not initialized yet. Use get_weights() method to initialize the weights"
            )

    def update_weights(self, weights: DeviceArray):
        if self.trainable:
            self.set_weights(weights)

    def __repr__(self) -> str:
        return f"{self.name} with shape {self.shape}>"


class NodeContainer:
    def __init__(self):
        self.inbound_nodes = []
        self.outbound_nodes = []

    def connect_nodes(self, current_node, connected_node):
        if isinstance(connected_node, (list, tuple)):
            for cn in connected_node:
                cn._node_container.outbound_nodes.append(current_node)
                current_node._node_container.inbound_nodes.append(cn)
        else:
            current_node._node_container.inbound_nodes.append(connected_node)
            connected_node._node_container.outbound_nodes.append(current_node)


class MetricsContainer:
    pass


# class KyosaiShapedArray(jax.core.ShapedArray):
#     def __init__(self, key, shape, initializer, dtype, name, trainable, weak_type=False, named_shape=None):
#         super().__init__(shape, dtype, weak_type=False, named_shape=None)
#         self.key = key
#         self.shape = shape
#         self.initializer = initializer
#         self.trainable = trainable
#         self.name = name
#         self._initalized = False

#     def initialize(self):
#         self.__class__ = Weight
