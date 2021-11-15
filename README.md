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


#This model is just a toy model for testing that everything works
inputs = Input((28, 28, 1))
conv1 = Conv2D(64, 3, activation=activations.ReLU, key=PRNGKey(100))(inputs)
act1 = Activation('relu')(conv1)
conv3 = Conv2D(128, 3, padding='same', key=PRNGKey(104))(act1)
act2 = Activation('relu')(conv3)
conv4 = Conv2D(128, 3, padding='same', key=PRNGKey(105))(act2)
flatten = Flatten()(conv4)
dense1 = Dense(512, activation='relu')(flatten)
dense2 = Dense(10, activation='softmax')(dense1)

model = kerax.Model(inputs, output)


model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(x_train, y_train, epochs=40, batch_size=64)

```
---


### Currently It's not fully ready, so there is no setup.py file
