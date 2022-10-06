# kyosai

Deep Learning library based on JAX and inspired from Keras



---
> Task List
- [ ] Add more layers
- [ ] Add documentation
- [ ] Add more loss functions
- [ ] Add more optimizers 
---


> Functional model example:

```python
import kyosai
from kyosai import layers
from jax import numpy as jnp

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
```


> Sequential model example:
```python
import kyosai
from kyosai import layers
from jax import numpy as jnp

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


> `Model` subclass example:
```python
import kyosai
from kyosai import layers
import numpy as np

class MyModel(kyosai.Model):
    def __init__(self):
        super().__init__()
        self.inputs = layers.Input((23, 128))
        self.conv1 = layers.Conv1D(32, 3, seed=7)
        self.maxpool1 = layers.MaxPooling1D(3, seed=7)
        self.conv2 = layers.Conv1D(64, 3, seed=7)
        self.mha = layers.MultiHeadAttention(64, 4, seed=7)
        self.maxpool2 = layers.GlobalMaxPooling1D()
        self.dense = layers.Dense(128, activation='relu', seed=7)
        self.out = layers.Dense(1, activation='sigmoid', seed=7)
    
    def call(self, x):
        x = self.inputs(x)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.mha(x, x, x)
        x = self.maxpool2(x)
        x = self.dense(x)
        x = self.out(x)
        return x

model = MyModel()
xs = np.random.random((64, 23, 128))
ys = np.random.random((64, 1))

model.compile(loss='binary_crossentropy', optimizer='adam')



```
---