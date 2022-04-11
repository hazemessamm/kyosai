# Kerax

Deep Learning library based on JAX and inspired from Keras



---
> Task List
- [ ] Add more layers
- [ ] Add documentation
- [ ] Add more loss functions
- [ ] Add more optimizers 
---

---

> Example:

```python
import kerax
from kerax import layers
from jax import numpy as jnp
from kerax.utils import to_categorical
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


# This model is just a toy model for demo.
inputs = layers.Input((28, 28, 1))
conv1 = layers.Conv2D(64, 3, activation=activations.ReLU, seed=100)(inputs)
act1 = layers.Activation('relu')(conv1)
conv2 = layers.Conv2D(128, 3, padding='same', seed=101)(act1)
act2 = layers.Activation('relu')(conv2)
conv3 = layers.Conv2D(128, 3, padding='same', seed=101)(act2)
flatten = layers.Flatten()(conv4)
dense1 = layers.Dense(512, activation='relu', seed=101)(flatten)
dense2 = layers.Dense(10, activation='softmax', seed=101)(dense1)

model = kerax.Model(inputs, output)


model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(x_train, y_train, epochs=40, batch_size=64)

```
---


## Currently, it's not fully ready, so there is no setup.py file.