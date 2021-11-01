from layers import Concatenate
from layers import Conv2D, Dense, Flatten, Input, BatchNormalization, Activation, Add
from models import Model
from losses import CategoricalCrossEntropy
import numpy as np
from copy import copy
train_x = np.random.random((128, 28,28,1))
train_y = np.random.random((128, 10))


val_x = np.random.random((128, 28,28,1))
val_y = np.random.random((128, 10))

inputs = Input((28, 28, 1))
x = Conv2D(64, 3)(inputs)
x = Activation('relu')(x)
x = Conv2D(128, 3)(x)
x = Activation('relu')(x)
res = Conv2D(128, 3, padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(128, 3, padding='same')(x)
x2 = copy(x)
x = Add([x, res])(x)
x1 = copy(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(10)(x)
outputs = Activation('softmax')(x)
model = Model(input=inputs, output=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.fit(train_x, train_y, epochs=10, batch_size=10, validation_data=(val_x, val_y))

print(model(train_x).shape)

print(x2.output)