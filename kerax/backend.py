import jax

from kerax.engine import Trackable

_PRECISION = 'float32'

def clear_session():
    Trackable.reset()

def current_platform():
    return jax.lib.xla_bridge.get_backend().platform

def is_gpu_available():
    return jax.lib.xla_bridge.get_backend().platform == 'gpu'

def devices():
    return jax.devices()

def device_count():
    return jax.device_count()

def backend():
    return 'jax'

def precision():
    global _PRECISION
    return _PRECISION

def set_precision(precision='float32'):
    global _PRECISION
    _PRECISION = precision


_IS_JIT_ENABLED = True

def enable_jit_execution(enable):
    global _IS_JIT_ENABLED
    if isinstance(enable, bool):
        _IS_JIT_ENABLED = enable
    else:
        raise TypeError(f'\"enable_jit\" should be boolean found type {type(enable)}')

def is_jit_enabled():
    global _IS_JIT_ENABLED
    return _IS_JIT_ENABLED
