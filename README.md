# Kerax

Deep Learning library based on JAX and inspired from Keras



---
> Task List
- [ ] Add more layers
- [ ] Add documentation
- [ ] Add loss functions
- [ ] Add more optimizers 
---

---

> Example:

```python
from layers import core
import models
from layers import convolutional as c
import numpy as np
from jax import numpy as jnp
from optimizers import optimizers
from utils import to_categorical
from jax import nn

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


inputs = core.Input((28,28,1))
conv1 = c.Conv2D(128,3, activation='relu')(inputs)
conv2 = c.Conv2D(64,3, activation='relu')(conv1)
flatten = core.Flatten()(conv2)
dense = core.Dense(512, activation='relu')(flatten)
output = core.Dense(10, activation='softmax')(dense)

model = models.Model(inputs, output)

def loss(params, x, y):
    preds = model.call_with_external_weights(x, params)
    return jnp.mean(-jnp.log(preds[y]))

model.compile(loss=loss, optimizer='adam')


model.fit(x_train, y_train, epochs=40, batch_size=64, validation_data=(x_test, y_test))

```
---


### Currently It's not fully ready, so there is no setup.py file
