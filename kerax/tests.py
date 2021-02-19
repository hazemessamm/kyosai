from __future__ import absolute_import
import set_path
from layers import core
import models
from layers import convolutional as c
import numpy as np
from optimizers import optimizers
from utils import to_categorical

###Testing
inputs = core.Input((64,28,28,1))
conv1 = c.Conv2D(128,3, activation='relu')(inputs)
conv2 = c.Conv2D(64,3, activation='relu')(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512, activation='relu')(flatten)
output = core.Dense(10)(dense)
output_activation = core.Activation('softmax')(output)

model = models.Model(inputs, output_activation)


def loss(params, x, y):
    preds = model.call_with_external_weights(x, params)
    return -np.sum(preds * y)

sgd = optimizers.Adam(loss_fn=loss, model=model)

model.compile(loss=loss, optimizer='sgd')


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype('float32')
x_test = np.expand_dims(x_test, -1)


y_train = to_categorical(y_train, 10)

model.fit(x_train, y_train, epochs=20, batch_size=64)