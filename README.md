# kyosai

Deep Learning library based on JAX and inspired from Keras



---
> Task List
- [ ] Add more layers
- [ ] Add documentation
- [ ] Add more loss functions
- [ ] Add more optimizers 
---


> Example:

```python
import kyosai
from kyosai import layers
from jax import numpy as jnp
from kyosai.utils import to_categorical


# You can define a Functional Model.
inputs = layers.Input(shape=(28, 28, 1))
conv1 = layers.Conv2D(64, 3, activation='relu')(inputs)
maxpool1 = layers.MaxPooling2D(2)(conv1)
conv2 = layers.Conv2D(64, 3, activation='relu')(maxpool1)
maxpool3 = layers.MaxPooling2D(2)(conv2)
flatten = layers.Flatten()(maxpool3)
dense1 = layers.Dense(128, activation='relu')(flatten)
output = layers.Dense(10, activation='softmax')(dense1)

model = kyosai.Model(inputs, output)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x_train, y_train, epochs=40, batch_size=64)


# Or you can define a Sequential Model
model = kyosai.Sequential([
  layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
  layers.MaxPooling2D(2),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(10)
])

model.compile(loss='categorical_crossentropy_from_logits', optimizer='adam')
model.fit(x_train, y_train, epochs=40, batch_size=64)

```
---