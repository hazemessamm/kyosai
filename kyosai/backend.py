import jax

_PRECISION = "float32"
_IS_JIT_ENABLED = True
_MEMO = {}


def memoize(input):
    global _MEMO
    _MEMO[input] = _MEMO.get(input, 0) + 1
    return f"{input}_{_MEMO[input]}"


def clear_session():
    _MEMO.clear()


def platform():
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
        raise TypeError(f"`enable_jit` should be " f"boolean. Recieved: {type(enable)}")


def is_jit_enabled():
    global _IS_JIT_ENABLED
    return _IS_JIT_ENABLED


def device_put(x, id):
    available_devices = devices()
    if len(available_devices) <= id:
        raise ValueError(
            f"Number of devices={len(available_devices)}. Recieved id={id}"
        )
    else:
        device = available_devices[id]
    return jax.device_put(x, device)


def get_model_gradients(model, loss):
    def _get_loss(params, x, y):
        preds = model.call_with_external_weights(params, x)
        loss_val = loss(y, preds)
        return (loss_val, preds)

    _get_grads = jax.value_and_grad(_get_loss, argnums=0, has_aux=True)

    def call(x, y):
        loss_and_preds, grads = _get_grads(model.params, x, y)
        return (*loss_and_preds, grads)

    return call
