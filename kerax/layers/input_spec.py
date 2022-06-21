import jax

# TODO:
class InputSpec:
    def __init__(self, input_shape):
        if input_shape[0] is None:
            self.input_shape = input_shape[1:]
        else:
            self.input_shape = input_shape
