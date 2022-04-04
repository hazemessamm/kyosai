from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kerax.initializers import Initializer, initializers

class Weight:
    def __init__(self, key: PRNGKey, shape: tuple, initializer: Initializer, dtype: str, name: str, trainable: bool):
        self.key = key
        self.shape = shape
        self.initializer = initializers.get(initializer)
        self.name = name
        self.trainable = trainable
        self.dtype = dtype
        self.built = False

    def get_weights(self, rebuild: bool = False):
        if self.built and not rebuild:
            return self.weights
        self.weights = self.initializer(self.key, self.shape, self.dtype)
        self.built = True
        return self.weights

    def set_weights(self, weights: DeviceArray):
        if self.built:
            if (self.weights.shape == weights.shape) and (self.dtype == weights.dtype):
                self.weights = weights
            else:
                raise Exception(f'New weights shape does not match the current weights shape, {self.weights.shape} != {weights.shape}')
        else:
            raise Exception(f'Weights are not initialized yet. Use get_weights() method to initialize the weights')


    def __repr__(self) -> str:
        return f'{self.name} with shape {self.shape}>'


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
