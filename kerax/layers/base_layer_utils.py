from kerax.layers import core

def is_in_construction_mode(inputs):
    return isinstance(inputs, core.Layer)