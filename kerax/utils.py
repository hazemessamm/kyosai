from numpy import max, arange

def to_categorical(inputs, num_classes=None):
    if num_classes is None:
        num_classes = max(inputs, axis=-1)
    return inputs[:, None] == arange(num_classes)