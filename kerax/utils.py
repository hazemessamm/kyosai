from numpy import max, arange

def to_categorical(inputs, num_classes=None):
    if num_classes is None:
        num_classes = max(inputs, axis=-1)
    return inputs[:, None] == arange(num_classes)



class Sequence:
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError