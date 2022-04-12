import jax

from kerax.engine import Trackable

_PRECISION = "float32"
_IS_JIT_ENABLED = True

def clear_session():
    Trackable.reset()

def current_platform():
    return jax.lib.xla_bridge.get_backend().platform

def is_gpu_available():
    return jax.lib.xla_bridge.get_backend().platform == "gpu"

def devices():
    return jax.devices()

def device_count():
    return jax.device_count()

def precision():
    global _PRECISION
    return _PRECISION

def set_precision(precision="float32"):
    global _PRECISION
    _PRECISION = precision

def enable_jit_execution(enable):
    global _IS_JIT_ENABLED
    if isinstance(enable, bool):
        _IS_JIT_ENABLED = enable
    else:
        raise TypeError(f'"enable_jit" should be boolean found type {type(enable)}')

def is_jit_enabled():
    global _IS_JIT_ENABLED
    return _IS_JIT_ENABLED

def device_put(x, id):
    available_devices = devices()
    if len(available_devices) <= id:
        raise ValueError(f'Number of devices={len(available_devices)}. Recieved index={id}')
    else:
        device = available_devices[id]
    return jax.device_put(x, device)
