from typing import Any, List, Tuple, Union

from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kyosai import activations, backend
from kyosai.engine.containers import NodeContainer, Weight
from kyosai.initializers import Initializer, initializers
from kyosai.layers import layer_utils
from numpy import ndarray
import abc
import inspect
from kyosai.engine import graph_recorder


class Layer(abc.ABC):
    def __init__(
        self,
        seed: int = None,
        trainable: bool = True,
        dtype: str = "float32",
        name: str = None,
    ):
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        layer_utils.jit_layer_call(self)

        # Stores the layer weights
        self._weights = []
        self._node_container = NodeContainer()
        self.seed = PRNGKey(layer_utils.check_seed(seed))
        self.trainable = trainable
        self.dtype = dtype or backend.precision()
        self.built = False
        self._has_nested_layers = False
        self._layers = []
        self._is_nested = False
        self._call_util = layer_utils.CallFunctionUtil(self.call)
        layer_utils.validate_layer_options(self)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Layer):
            if not self._has_nested_layers:
                self._has_nested_layers = True
                self._layers = [__value]
            else:
                self._layers.append(__value)
            __value._is_nested = True
        return super().__setattr__(__name, __value)

    @property
    def nested_layers(self):
        return self._layers

    @property
    def shape(self):
        return None

    @property
    def parents(self):
        return self._node_container.inbound_nodes

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        raise NotImplementedError(
            f"Error in layer {self}. Should be implemented in a subclass."
        )

    @property
    def input_shape(self):
        if self.built:
            return self._input_shape
        raise Exception(f"Error in {self.name}, Layer is not built yet")

    @property
    def weights(self):
        if not self.built:
            raise Exception(
                f"Error in layer {self}. Layer is not built yet. use `build()` method."
            )
        weights = [weight.get_weights() for weight in self._weights]
        if self._has_nested_layers:
            nested_weights = tuple(layer.weights for layer in self._layers)
            weights.extend(nested_weights)
        return tuple(weights)

    @property
    def named_weights(self):
        return {weight.name: weight.get_weights() for weight in self._weights}
    
    @property
    def trainable_weights(self):
        if not self.built:
            raise Exception(
                f"Error in layer {self}. Layer is not built yet. use `build()` method."
            )

        weights = [weight.get_weights() if weight.trainable else () for weight in self._weights]
        if self._has_nested_layers:
            nested_weights = tuple(layer.trainable_weights for layer in self._layers)
            weights.extend(nested_weights)
        return tuple(weights)

    def build(self, input_shape: Tuple):
        raise NotImplementedError(
            f"Error in layer {self}. Should be implemented in a subclass."
        )

    def get_initializer(self, identifier: Union[str, Initializer]):
        "Returns the specified initializer"
        return initializers.get(identifier)

    def get_activation(self, identifier: Union[str, Initializer]):
        "Returns the specified activation"
        return activations.get(identifier)

    def connect(self, layer, *args, **kwargs):
        "Connects the current layer with the previous layer"

        if isinstance(layer, graph_recorder.GraphRecorder):
            layer = layer.from_layer
        self._node_container.connect_nodes(self, layer)

        for arg in args:
            if isinstance(arg, Layer):
                self._node_container.connect_nodes(self, arg)
            elif isinstance(arg, graph_recorder.GraphRecorder):
                self._node_container.connect_nodes(self, arg.from_layer)
            else:
                raise ValueError(
                    f"Error in layer {self}. `{arg}` is not a subclass of a layer. Recieved type={type(arg)}"
                )

        for k, v in kwargs.items():
            if isinstance(v, Layer):
                self._node_container.connect_nodes(self, v)
            elif isinstance(arg, graph_recorder.GraphRecorder):
                self._node_container.connect_nodes(self, arg.from_layer)
            else:
                raise ValueError(
                    f"Error in layer {self}. `{k}` is not a subclass of a layer. Recieved type={type(v)}"
                )

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
        self._weights.append(weight)
        return weight

    def get_weights(self):
        return self._weights

    def set_weights(self, new_weights: Tuple):
        if len(new_weights) > 0:
            if self._has_nested_layers:
                for layer in self._layers:
                    for w1, w2 in zip(layer._weights, new_weights):
                        w1.set_weights(w2)
            else:
                for w1, w2 in zip(self._weights, new_weights):
                    w1.set_weights(w2)

    def update_weights(self, new_weights: Tuple):
        if len(new_weights) > 0:
            for w_old, w_new in zip(self._weights, new_weights):
                w_old.update_weights(w_new)

            if self._has_nested_layers:
                for layer, new_weight in zip(self._layers, new_weights):
                    for w_old, w_new in zip(layer._weights, new_weight):
                        w_old.update_weights(w_new)

    def check_shape_if_built(self, layers):
        if isinstance(layers, (list, tuple)):
            for layer in layers:
                if self.input_shape != layer.shape:
                    raise ValueError(
                        f"Error in layer {self}"
                        f"This input shape of layer {self.name} does not match the output shape of layer {layer.name}."
                        f"Expected: {self.input_shape}. Recieved: {layer.shape}"
                    )
        else:
            if self.input_shape != layers.shape:
                raise ValueError(
                    f"Error in layer {self}"
                    f"This input shape of layer {self.name} does not match the output shape of layer {layers.name}."
                    f"Expected: {self.input_shape}. Recieved: {layers.shape}"
                )

    def parse_for_build(self, inputs, *args, **kwargs):
        fn_args = inspect.getfullargspec(self.build).args
        if isinstance(inputs, list):
            input_shape = [getattr(i, "shape", None) for i in inputs]
        else:
            input_shape = inputs.shape

        if (
            not isinstance(input_shape, list) and not getattr(inputs, "shape", None)
        ) or (isinstance(input_shape, list) and not all(input_shape)):
            raise Exception(
                f"Error in layer {self}. `inputs` does not have `shape` attribute."
            )
        arg_shapes = []
        for arg, fn_arg in zip(args, fn_args):
            if not getattr(arg, "shape", None):
                raise Exception(
                    f"Error in layer {self}. `{fn_arg}` does not have `shape` attribute."
                )
            arg_shapes.append(arg.shape)

        kwarg_shapes = {}
        for k, w in kwargs.items():
            if getattr(w, "shape", None):
                raise Exception(
                    f"Error in layer {self}. `{k}` does not have `shape` attribute."
                )
            kwarg_shapes[k] = w.shape

        return input_shape, arg_shapes, kwarg_shapes

    def dummy_call(self, inputs, *args, **kwargs):
        if inputs.latest_layer is not None:
            inputs_shape, args_shape, kwargs_shape = self.parse_for_build(
                inputs, *args, **kwargs
            )
            self.build(inputs_shape, *args_shape, **kwargs_shape)
            self.connect(inputs.latest_layer, *args, **kwargs)
        inputs.latest_shape = self.shape
        inputs.latest_layer = self
        return inputs

    def build_for_layer(self, inputs, *args, **kwargs):
        # temporary and will be removed or modified
        inputs_shape, args_shape, kwargs_shape = self.parse_for_build(
            inputs, *args, **kwargs
        )
        self.build(inputs_shape, *args_shape, **kwargs_shape)
        self.connect(inputs, *args, **kwargs)
        return self

    def build_and_call_for_tensors(self, inputs, *args, **kwargs):
        # temporary and will be removed or modified
        inputs_shape, args_shape, kwargs_shape = self.parse_for_build(
            inputs, *args, **kwargs
        )
        self.build(inputs_shape, *args_shape, **kwargs_shape)
        output = self.call(inputs, *args, **kwargs)
        return output

    def build_for_list(self, inputs, *args, **kwargs):
        # temporary and will be removed or modified
        inputs_shape, args_shape, kwargs_shape = self.parse_for_build(
            inputs, *args, **kwargs
        )
        self.build(inputs_shape, *args_shape, **kwargs_shape)

    def __call__(self, *args, **kwargs):
        inputs, args, kwargs = self._call_util.parse_args(*args, **kwargs)
        if isinstance(inputs, graph_recorder.GraphRecorder):
            return self.dummy_call(inputs, *args, **kwargs)

        if not self.built:
            if isinstance(inputs, (Layer)):
                return self.build_for_layer(inputs, *args, **kwargs)
            elif isinstance(inputs, (ndarray, DeviceArray, DynamicJaxprTracer)):
                return self.build_and_call_for_tensors(inputs, *args, **kwargs)
            elif isinstance(inputs, (list, tuple)):
                self.build_for_list(inputs, *args, **kwargs)
                if all([isinstance(i, Layer) for i in inputs]):
                    self.connect(inputs, *args, **kwargs)
                    return self
                else:
                    return self.call(inputs, *args, **kwargs)
            else:
                raise ValueError(
                    f"Error in layer {self}. "
                    f"`inputs` should be with type `Layer`, `ndarray`, `DeviceArray`, `list` or `tuple`."
                    f"Recieved: {type(inputs)}"
                )
        else:
            if isinstance(inputs, Layer):
                self.check_shape_if_built(inputs)
                self.connect(inputs, *args, **kwargs)
                return self
            else:
                return self.call(inputs, *args, **kwargs)

    @abc.abstractmethod
    def call(self, inputs: DeviceArray, weights: Tuple = None, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    def __repr__(self):
        if self.built and getattr(self, 'input_shape', None):
            return f"<{self.name} Layer with input shape {self.input_shape} and output shape {self.shape}>"
        else:
            return f"<{self.name} Layer>"

    def __name__(self):
        return self.name
