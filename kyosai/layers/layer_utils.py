import inspect

import numpy as np
from jax import jit
from kyosai import backend


def check_seed(seed: int):
    if seed is None:
        return np.random.randint(1e6)
    elif not isinstance(seed, int):
        raise ValueError(
            f"`seed` should be with type int. Recieved: seed={seed} with type={type(seed)}"
        )
    else:
        return seed


def jit_layer_call(layer):
    if backend.is_jit_enabled():
        layer.call = jit(layer.call)
        layer.call_with_external_weights = jit(layer.call_with_external_weights)


def validate_layer_options(layer):
    call_with_external_weights_weights = inspect.getfullargspec(
        layer.call_with_external_weights
    ).args
    if call_with_external_weights_weights[0] == "self":
        call_with_external_weights_weights = call_with_external_weights_weights[1:]
    if call_with_external_weights_weights[0] != "weights":
        raise ValueError(
            f"`weights` argument should be added as the first argument in `call_with_external_weights` function."
            f"Recieved: {call_with_external_weights_weights}"
        )


class CallFunctionUtil:
    def __init__(self, func):
        self._func_specs = inspect.getfullargspec(func)
        self._arg_names = self._func_specs.args + (self._func_specs.kwonlyargs or [])

        if self._arg_names[0] == "self":
            self._arg_names = self._arg_names[1:]

        self._num_args = len(self._arg_names)
        self._has_training_arg = "training" in self._arg_names
        self._has_mask_arg = "mask" in self._arg_names
        self._accept_kwargs = self._func_specs.varkw
        self._requires_unpacking = None

    # TOOD: try to enhance the `parse` method.
    def parse_experimental(self, *args, **kwargs):
        pass

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

    def parse_args(self, *args, **kwargs):
        if args:
            inputs = args[0]
            args = args[1:]
        else:
            if self._arg_names[0] in kwargs:
                inputs = kwargs.pop(self._arg_names[0])
        if self._num_args > 1:
            self._requires_unpacking = True
        return inputs, args, kwargs
