from jax import jit
from kerax import backend


def flatten(x):
    if not isinstance(x, (list, tuple)):
        return [x]

    def _flatten(x, result=[]):
        for i in x:
            if isinstance(i, list):
                return _flatten(i, result)
            else:
                result.append(i)
        return result

    return _flatten(x, [])


def jit_call(graph):
    if backend.is_jit_enabled():
        graph.call_with_external_weights = jit(graph.call_with_external_weights)
