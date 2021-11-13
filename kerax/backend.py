from kerax.engine import Trackable
import jax

PRECISION = 'float32'

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
    global PRECISION
    return PRECISION

def set_precision(precision='float32'):
    global PRECISION
    PRECISION = precision
