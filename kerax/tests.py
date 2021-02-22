from __future__ import absolute_import
#import set_path
from layers import core
import models
from layers import convolutional as c
import numpy as np
from jax import numpy as jnp
from optimizers import optimizers
from utils import to_categorical
from jax.scipy.special import logsumexp
from jax import nn
from jax import random


###Testing
inputs = core.Input((64,28,28,1))
conv1 = c.Conv2D(64,3, activation='relu', key=random.PRNGKey(1003))(inputs)
conv2 = c.Conv2D(64,3, activation='relu')(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512, activation='relu')(flatten)
output = core.Dense(10, activation='softmax')(dense)

model = models.Model(inputs, output)

def loss(model):
    def loss_fn(params, x, y):
        preds = model.call_with_external_weights(x, params)
        return jnp.mean(-jnp.log(preds[y]))
    return loss_fn

def loss_v2(params, x, y):
    preds = model.call_with_external_weights(x, params)
    return jnp.mean(-jnp.log(preds[y]))

model.compile(loss=loss_v2, optimizer='adam')



from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255.0
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype('float32')
x_test = np.expand_dims(x_test, -1)
y_train = y_train.astype('float32')

y_train = to_categorical(y_train, 10)

sample = x_test[0]
sample_x = jnp.expand_dims(sample, 0)
sample_y = y_test[0]


model.fit(x_train, y_train, epochs=40, batch_size=64, validation_data=(x_test, y_test))

print(jnp.argmax(model(sample_x)))


'''
Tests with sequential model

import numpy as np

def loss(params, x, y):
    preds = model.call_with_external_weights(x, params)
    return jnp.mean(-jnp.log(preds[y]))


layers = [core.Input((28, 28, 1)), cl.Conv2D(3,3), cl.Conv2D(3,3), core.Flatten()]
model = models.Sequential()
model.add(core.Input((28,28,1)))
model.add(cl.Conv2D(3,3))
model.add(cl.Conv2D(3,3))
model.add(core.Flatten())
model.add(core.Dense(10, activation='softmax'))
model.compile(loss=loss, optimizer='adam')

x = np.random.random((64, 28,28,1))
y = [np.random.randint(11) for i in range(64)]
y = np.array(y)
from utils import to_categorical, to_numbers
y = to_categorical(y)


model.fit(x, y)

'''
