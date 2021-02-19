import set_path
from layers import core
from layers import convolutional as c

class Model:
    def __init__(self, input_layer, output_layer, name=None):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.name = name if name is not None else self.__class__.__name__
        self.layers = []
        self.trainable_params = []
        self.get_layers_and_params()
    
    def get_layers_and_params(self):
        pointer = self.output_layer
        while hasattr(pointer, 'prev'):
            self.layers.append(pointer)
            self.trainable_params.append(pointer.get_weights())
            pointer = pointer.prev
        #Appending Input Layer
        self.layers.append(pointer)

    def compile(self, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer
    
        



###Testing
inputs = core.Input((64,28,28,1))
conv1 = c.Conv2D(3,3)(inputs)
conv2 = c.Conv2D(3,3)(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512)(flatten)
output = core.Dense(10)(dense)
output_activation = core.Activation('softmax')(output)

model = Model(inputs, output_activation)
print(model.trainable_params)