import abc



class DataAdapter(metaclass=abc.ABCMeta):
    def __init__(self, x, y=None, **kwargs):
        self.x = x
        self.y = y
    
    @abc.abstractmethod
    def get_dataset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def batch_size(self):
        raise NotImplementedError
    

class TensorLikeDataAdapter(DataAdapter):
    def __init__(self, x, y=None, batch_size=None, epochs=1, steps=None, shuffle=False, **kwargs):
        super(TensorLikeDataAdapter, self).__init__(x, y, **kwargs)

        if steps is None:
            steps = 32
        if batch_size is None:
            num_samples = x.shape[0]
            batch_size = num_samples // steps
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.shuffle = shuffle

    def get_batch(self):
        pass
    
    
