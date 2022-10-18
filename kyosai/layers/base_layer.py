from typing import Any, List, Tuple, Union
import jax
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.numpy import DeviceArray
from jax.numpy import ndarray as jax_ndarray
from numpy import ndarray as numpy_ndarray
from jax.random import PRNGKey
from kyosai import activations, backend
from kyosai.engine.containers import NodeContainer, Weight
from kyosai.initializers import Initializer, initializers
from kyosai.layers import layer_utils
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
        **kwargs,
    ):
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)

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
        self._in_eval_mode = False
        self._input_shape = None
        self._output_shape = None
        self._call_util = layer_utils.CallFunctionUtil(self.call)

        input_shape = kwargs.get("input_shape", None)

        if input_shape:
            self.input_shape = input_shape

        # layer_utils.jit_layer_call(self)
        layer_utils.validate_layer_options(self)

    def track_nested_layer(self, layer):
        if not self._has_nested_layers:
            self._has_nested_layers = True

        self._layers.append(layer)
        layer._is_nested = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Layer):
            self.track_nested_layer(__value)
        return super().__setattr__(__name, __value)

    @property
    def input_shape(self):
        if self._input_shape is None:
            return None

        num_build_args = len(inspect.getfullargspec(self.build).args[1:])
        if num_build_args > 1 and not isinstance(self._input_shape, list):
            return [self._input_shape]
        return self._input_shape

    @input_shape.setter
    def input_shape(self, val):
        if isinstance(val, list):
            self._input_shape = [(None, *v) if v[0] is not None else v for v in val]
        else:
            if self._input_shape is not None and self._input_shape[0] is not None:
                self.input_shape = (None, *self._input_shape)
            elif self._input_shape is None and val[0] is not None:
                self._input_shape = (None, *val)
            else:
                self._input_shape = val

    @property
    def nested_layers(self):
        return self._layers

    @property
    def shape(self):
        if self.input_shape is None:
            raise Exception(
                f"Error in layer {self}. This layer does not have input_shape. Build it, pass an `Input` instance to it or pass `input_shape` to `__init__`."
            )
        if self._output_shape is not None:
            return self._output_shape

        self._output_shape = self.evaluate_forward_methods(self.input_shape)

        num_build_args = len(inspect.getfullargspec(self.build).args[1:])
        if num_build_args > 1 and not isinstance(out_shape, list):
            out_shape = [out_shape]
            self._output_shape = [
                (None, *v[1:]) if v[0] is not None else v for v in out_shape
            ]
        else:
            if self._output_shape[0] is not None:
                self._output_shape = (None, *self._output_shape[1:])

        return self._output_shape

    @property
    def parents(self):
        return self._node_container.inbound_nodes

    def compute_output_shape(self, input_shape):
        raise NotImplementedError(
            "`compute_output_shape` method should be implemented in a subclass."
        )

    def evaluate_forward_methods(self, input_shape: List):
        "Computes the output shape given the specified arguments when instantiated."
        try:
            output_shape = self.compute_output_shape(input_shape=input_shape)
            return output_shape
        except NotImplementedError:
            pass

        dummy_batch_size = 1
        if isinstance(input_shape, list):
            inputs = []
            for inp_shape in input_shape:
                if inp_shape[0] is None:
                    current_shape = (dummy_batch_size, *inp_shape[1:])
                else:
                    current_shape = (dummy_batch_size, *inp_shape)
                inputs.append(
                    jax.core.ShapedArray(shape=current_shape, dtype=self.dtype)
                )
        else:
            if input_shape[0] is None:
                inputs = (dummy_batch_size, *input_shape[1:])
                inputs = jax.core.ShapedArray(shape=inputs, dtype=self.dtype)
            else:
                inputs = jax.core.ShapedArray(shape=input_shape, dtype=self.dtype)

        self._in_eval_mode = True
        call_shape = jax.eval_shape(self.call, inputs).shape
        call_with_weights_shape = jax.eval_shape(
            self.call_with_external_weights, self.weights, inputs
        ).shape
        self._in_eval_mode = False
        if call_shape != call_with_weights_shape:
            raise Exception(
                f"`call` method output shape does not match `call_with_external_weights` output shape. {call_shape} != {call_with_weights_shape}."
            )
        else:
            return call_shape

    def _get_weights(self, trainable_only=False):
        if not self.built:
            self.build(self.input_shape)
        elif not self._in_eval_mode:
            if any(isinstance(w.weights, jax.core.Tracer) for w in self._weights):
                self._weights = []
                self.build(self.input_shape)

        if not trainable_only:
            weights = [weight.get_weights() for weight in self._weights]
            if self._has_nested_layers:
                nested_weights = tuple(layer.weights for layer in self._layers)
                weights += nested_weights
        else:
            weights = [
                weight.get_weights() if weight.trainable else ()
                for weight in self._weights
            ]

        if self._has_nested_layers:
            nested_weights = tuple(layer.trainable_weights for layer in self._layers)
            weights += nested_weights
        weights = tuple(weights)
        return weights

    @property
    def weights(self):
        weights = self._get_weights(trainable_only=False)
        self._weights_instantiated = True
        return weights

    @property
    def trainable_weights(self):
        weights = self._get_weights(trainable_only=True)
        return weights

    def build(self, input_shape: Tuple):
        self.input_shape = input_shape
        self.built = True

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
        fn_args = inspect.getfullargspec(self.build).args[1:]
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
        if all([isinstance(i, Layer) for i in inputs]):
            self.connect(inputs, *args, **kwargs)
            return self
        else:
            return self.call(inputs, *args, **kwargs)

    def __call__(self, *args, **kwargs):

        # cases:
        # 1. if a model subclassed and should traverse through call function
        # 2. if the input is a layer\s
        # a. it can be 1 layer, list of layers, multiple inputs as layers.
        # 3. if the input is a tensor\s
        # 4. if the layer is already built but will pass to it another layer
        # 5. if a model is passed
        inputs, args, kwargs = self._call_util.parse_args(*args, **kwargs)

        if isinstance(inputs, graph_recorder.GraphRecorder):
            return self.dummy_call(inputs, *args, **kwargs)

        if not self.built:
            if isinstance(inputs, (Layer)):
                return self.build_for_layer(inputs, *args, **kwargs)
            elif isinstance(
                inputs, (jax_ndarray, numpy_ndarray, DeviceArray, DynamicJaxprTracer)
            ):
                return self.build_and_call_for_tensors(inputs, *args, **kwargs)
            elif isinstance(inputs, (list, tuple)):
                return self.build_for_list(inputs, *args, **kwargs)
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
    def call(self, inputs: DeviceArray, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    @abc.abstractmethod
    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    def __repr__(self):
        if self._input_shape is not None and self._output_shape is not None:
            return f"<{self.name} Layer with input shape {self.input_shape} and output shape {self.shape[0]}>"
        else:
            return f"<{self.name} Layer>"

    def __name__(self):
        return self.name
