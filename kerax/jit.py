


IS_JIT_ENABLED = True

def enable_jit_execution(enable_jit):
    global IS_JIT_ENABLED
    if isinstance(enable_jit, bool):
        IS_JIT_ENABLED = enable_jit
    else:
        raise TypeError(f'\"enable_jit\" should be boolean found type {type(enable_jit)}')