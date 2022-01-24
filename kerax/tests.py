import numpy as np
from jax import numpy as jnp
from jax import random

import layers
import losses
import models
import optimizers
from utils import to_categorical

###Testing
inputs = layers.Input((64,28,28,1))
conv1 = layers.Conv2D(64,3, activation='relu', key=random.PRNGKey(1003))(inputs)
conv2 = layers.Conv2D(128,3, activation='relu', key=random.PRNGKey(1005))(conv1)
flatten = layers.Flatten()(conv2)
dense = layers.Dense(512, activation='relu', key=random.PRNGKey(1006))(flatten)
dense2 = layers.Dense(10, key=random.PRNGKey(1008))(dense)
outputs = layers.Activation('softmax')(dense2)

model = models.Model(input=inputs, output=outputs)


model.compile(loss=losses.CategoricalCrossEntropy(model), optimizer='adam')


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_train /= 255.0
x_train = np.expand_dims(x_train, -1)
x_test = x_test.astype('float32')
x_test /= 255.0
x_test = np.expand_dims(x_test, -1)
y_train = y_train.astype('float32')

y_train = to_categorical(y_train, 10)

sample = x_test[0]
sample_x = jnp.expand_dims(sample, 0)
sample_y = y_test[0]


model.fit(x_train, y_train, epochs=1, batch_size=128)
