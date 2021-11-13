
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