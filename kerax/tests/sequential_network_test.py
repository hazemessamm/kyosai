import kerax
from kerax import layers
import numpy as np


# Sequential Model
model = kerax.Sequential(
    [
        layers.Conv2D(64, 3, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)


X = np.random.random((256, 28, 28, 1))
y = kerax.utils.to_categorical(np.random.randint(1, 11, (256,)))


model.fit(X, y)
