import inspect
from typing import Any, List, Tuple, Union

from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.numpy import DeviceArray
from jax.random import PRNGKey
from kerax import activations, backend
from kerax.layers import layer_utils
from kerax.engine.containers import NodeContainer, Weight
from kerax.initializers import Initializer, initializers
from numpy import ndarray


class CallFunctionUtil:
    def __init__(self, func):
        self._func_specs = inspect.getfullargspec(func)
        self._arg_names = self._func_specs.args + (self._func_specs.kwonlyargs or [])
        self._num_args = len(self._arg_names)
        self._has_training_arg = "training" in self._arg_names
        self._has_mask_arg = "mask" in self._arg_names
        self._accept_kwargs = self._func_specs.varkw
        self._requires_unpacking = None

    def parse(self, *args, **kwargs):
        inputs = []
        extra_args = []
        if args:
            for arg in args:
                if hasattr(arg, "shape"):
                    inputs.append(arg)
                else:
                    extra_args.append(arg)
        else:
            for argname in self._arg_names:
                current_input = kwargs.pop(argname, False)
                if hasattr(current_input, "shape"):
                    inputs.append(current_input)

        if self._requires_unpacking is None:
            if len(inputs) == 1:
                self._requires_unpacking = False
                inputs = inputs[0]
            else:
                self._requires_unpacking = True
        else:
            if not self._requires_unpacking:
                inputs = inputs[0]

        return inputs, extra_args, kwargs

    def parse_v2(self, *args, **kwargs):
        if args:
            inputs = args[0]
            args = args[1:]
        else:
            if self._arg_names[0] in kwargs:
                inputs = kwargs.pop(self._arg_names[0])
        return inputs, args, kwargs


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
        self._node_container = NodeContainer()
        self.seed = PRNGKey(layer_utils._check_seed(seed))
        self.trainable = trainable
        self.dtype = dtype or backend.precision()
        self.built = False
        self._has_nested_layers = False
        self._is_nested = False
        self._validated = False
        self._call_util = CallFunctionUtil(self.call)
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
                f"`params` argument should be added as the first argument in `call_with_external_weights` function."
                f"Recieved: {call_with_external_weights_params.keys()}"
            )

    @property
    def shape(self):
        return None

    @property
    def parents(self):
        return self._node_container.inbound_nodes

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        raise NotImplementedError("should be implemented in a subclass.")

    @property
    def input_shape(self):
        if self.built:
            return self._input_shape
        raise Exception(f"Error in {self.name}, Layer is not built yet")

    @property
    def weights(self):
        "returns weights"
        return self._params

    @property
    def params(self):
        if not self.built:
            raise Exception("Layer is not built yet. use `build()` method.")
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
            for layer in layers:
                if self.input_shape != layer.shape:
                    raise ValueError(
                        f"This input shape of layer {self.name} does not match the output shape of layer {layer.name}."
                        f"Expected: {self.input_shape}. Recieved: {layer.shape}"
                    )
        else:
            if self.input_shape != layers.shape:
                raise ValueError(
                    f"This input shape of layer {self.name} does not match the output shape of layer {layers.name}."
                    f"Expected: {self.input_shape}. Recieved: {layers.shape}"
                )

    def __call__(self, *args, **kwargs):
        inputs, args, kwargs = self._call_util.parse_v2(*args, **kwargs)
        if not self.built:
            if isinstance(inputs, Layer):
                self.build(inputs.shape)
                self.connect(inputs)
                return self
            elif isinstance(inputs, (ndarray, DeviceArray, DynamicJaxprTracer)):
                self.build(inputs.shape)
                return self.call(inputs, *args, **kwargs)
            elif isinstance(inputs, (list, tuple)):
                shapes = [i.shape for i in inputs]
                self.build(shapes)
                if all([isinstance(i, Layer) for i in inputs]):
                    self.connect(inputs)
                    return self
                else:
                    return self.call(inputs, *args, **kwargs)
            else:
                raise ValueError(
                    f"`inputs` should be with type `Layer`, `ndarray`, `DeviceArray`, `list` or `tuple`."
                    f"Recieved: {type(inputs)}"
                )
        else:
            if isinstance(inputs, Layer):
                self.check_shape_if_built(inputs)
                self.connect(inputs)
                return self
            else:
                return self.call(inputs, *args, **kwargs)

    def call(self, inputs: DeviceArray, **kwargs):
        raise NotImplementedError(
            "This method should be implemented in Layer subclasses"
        )

    def call_with_external_weights(self, params: Tuple, inputs: DeviceArray, **kwargs):
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
