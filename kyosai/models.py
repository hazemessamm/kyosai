from kyosai.engine import model
from kyosai.engine.graph_v3 import GraphV3
from kyosai.layers.core import Input, Layer


def all_input_instances(arg):
    return all([isinstance(x, Input) for x in arg])


def all_layer_instances(arg):
    return all([isinstance(x, Layer) for x in arg])


def is_functional_params(*args, **kwargs):
    is_functional = False
    for i, arg in enumerate(args):
        if isinstance(arg, (list, tuple)):
            is_functional = (
                all_input_instances(arg) if i == 0 else all_layer_instances(arg)
            )
        else:
            if i == 0 and isinstance(arg, Input):
                is_functional = True
            elif isinstance(arg, Layer):
                is_functional = True

    for i, arg in enumerate(kwargs.values()):
        if isinstance(arg, (list, tuple)):
            is_functional = (
                all_input_instances(arg) if i == 0 else all_layer_instances(arg)
            )
        else:
            if i == 0 and isinstance(arg, Input):
                is_functional = True
            elif isinstance(arg, Layer):
                is_functional = True
    return is_functional


class Model(model._Model):
    def __new__(cls, *args, **kwargs):
        if is_functional_params(*args, **kwargs):
            return GraphV3(*args, **kwargs)
        else:
            return super(Model, cls).__new__(cls)
