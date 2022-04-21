from __future__ import absolute_import

import jax
from tqdm import trange

from kerax import losses
from kerax import metrics as _metrics
from kerax import optimizers
from kerax.engine import Trackable
from kerax.engine import data_adapter
from kerax.engine.graph import GraphV2
from kerax.layers.core import Layer
from typing import Any


def is_functional_params(*args, **kwargs):
    is_functional = False
    for arg in args:
        if isinstance(arg, Layer):
            is_functional = True

    for val in kwargs.values():
        if isinstance(val, Layer):
            is_functional = True

    return is_functional


class Model(Trackable):
    def __new__(cls, *args, **kwargs):
        if cls == Model and is_functional_params(*args, **kwargs):
            return GraphV2(*args, **kwargs)
        else:
            super(Model, cls).__new__(cls, *args, **kwargs)


class Model(Trackable):
    """
    Model class
    input_layer: takes the input layer
    output_layer: takes the output layer
    """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(
            self.__class__.__name__
            if kwargs.get("name") is None
            else kwargs.get("name")
        )
        # Aliases
        self.predict = self.__call__
        self.predict_with_external_weights = self.call_with_external_weights
        self._built = False
        self.trainable = kwargs.get("trainable", True)
        self._training_phase = False
        self._compiled = False
        self._sequential_model = isinstance(self, Sequential)
        self.metrics_instances = {}

        if len(type(self).__mro__) == 3:
            self.initialize_graph(*args, **kwargs)

    def __getattribute__(self, __name: str):
        if __name == "compile":
            self._compiled = True
            self._metrics_values = {}
        return super().__getattribute__(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, losses.Loss):
            __value.prediction_fn = self.call_with_external_weights

        return super().__setattr__(__name, __value)

    @property
    def __name__(self):
        return self.name

    @property
    def layers(self):
        if self._sequential_model:
            return self._layers
        return self.graph.layers.values()

    @property
    def params(self):
        if self._sequential_model:
            return self._params
        return self.graph.params

    @property
    def weights(self):
        if self._sequential_model:
            return self._params
        return self.graph.params

    @property
    def compiled(self):
        return self._compiled

    def initialize_graph(self, *args, **kwargs):
        "Stores the layers and paramters"
        if not self._sequential_model:
            self.graph = GraphV2(*args, **kwargs)

    def _get_metrics(self, metrics):
        if metrics is not None:
            if not isinstance(metrics, list):
                raise ValueError(
                    f"metrics should be inside a list. Recieved: {metrics}"
                )
            else:
                for metric in metrics:
                    if isinstance(metric, str):
                        self.metrics_instances[metric] = _metrics.get(metric)()
                    else:
                        self.metrics_instances[
                            metric.__class__.__name__
                        ] = _metrics.get(metric)

    def compile(self, loss, optimizer, metrics=None):
        "Takes the loss, optimizer and loss recorder state"
        self.loss_fn = losses.get(loss)
        self.optimizer = optimizers.get(optimizer)

        if isinstance(self.loss_fn, type):
            self.loss_fn = self.loss_fn()

        if isinstance(self.optimizer, type):
            self.optimizer = self.optimizer()

        self.optimizer.initialize(self.params)
        self._get_metrics(metrics)

    def __call__(self, x, training=False):
        "Takes inputs and returns predictions"
        return self.graph(x)

    def call_with_external_weights(self, params, x):
        "Takes inputs and params and returns predictions"
        return self.graph.call_with_external_weights(params, x)

    def get_weights(self):
        if self._sequential_model:
            return self._params
        return self.graph.params

    def set_weights(self, weights):
        "Set new weights on every layer"
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self.graph.params = weights

    def update_weights(self, updated_weights):
        for layer, w in zip(self.layers, updated_weights):
            layer.update_weights(w)
        self.graph.params = updated_weights

    def train_step(self, x, y):
        "Returns loss value and takes training batch"
        (loss, predictions), gradients = self.loss_fn(
            self.params, x, y, return_gradients=True
        )
        params = self.optimizer.minimize(self.params, gradients)
        self.update_weights(params)
        self._metrics_values.update({"loss": loss})
        return predictions

    def test_step(self, validation_dataset):
        avg_valid_loss = 0
        for _ in range(validation_dataset.num_batches):
            batch_x, batch_y = validation_dataset.get_batch()
            avg_valid_loss += self.loss_fn(batch_x, batch_y, return_gradients=False)
        self._metrics_values.update({"Validation loss": avg_valid_loss})

    def _test_step(self, validation_dataset):
        if validation_dataset is None:
            return {}
        else:
            return self.test_step(validation_dataset)

    def fit(
        self,
        x,
        y,
        epochs=1,
        batch_size=None,
        steps=None,
        shuffle=True,
        validation_data=None,
    ):
        if not self.compiled:
            raise Exception("Model is not compiled, use compile() method")

        dataset = data_adapter.TensorLikeDataAdapter(
            x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=shuffle
        )

        if validation_data:
            validation_dataset = data_adapter.TensorLikeDataAdapter(
                x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=False
            )
        else:
            validation_dataset = None

        for epoch in range(1, epochs + 1):
            self._metrics_values.clear()
            with trange(
                len(dataset),
                bar_format="{l_bar}{bar:40}{r_bar}{bar:-20b}",
                ascii=" =",
                unit="batch",
            ) as prog_bar:
                for _ in prog_bar:
                    batch_x, batch_y = dataset.get_batch()
                    predictions = self.train_step(batch_x, batch_y)
                    self._test_step(validation_dataset)

                    for metric_name, metric_instance in self.metrics_instances.items():
                        self._metrics_values.update(
                            {metric_name: metric_instance(batch_y, predictions)}
                        )

                    prog_bar.set_postfix(**self._metrics_values)


class Sequential(Model):
    def __init__(self, layers=None, **kwargs):
        super().__init__(**kwargs)

        self._layers = layers if layers is not None else []
        self._params = []
        self.validate_init()

    def validate_init(self):
        if self._layers is not None:
            self.connect_layers()

    def connect_layers(self):
        for i in range(len(self.layers) - 1):
            self._layers[i + 1](self.layers[i])

    def add(self, layer):
        if isinstance(layer, Layer):
            if len(self._layers) >= 1:
                layer(self._layers[-1])
            self._layers.append(layer)
            self.params.append(layer.params)
        else:
            raise ValueError(
                f"add() only accepts layers subclass instances or Input instance. Recieved layer={layer}"
            )

    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def call_with_external_weights(self, params, inputs):
        "Takes inputs and params and returns predictions"
        outputs = inputs
        for param, layer in zip(params, self.layers):
            outputs = layer.call_with_external_weights(param, outputs)
        return outputs

    def set_weights(self, weights):
        "Set new weights on every layer"
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self._params = weights

    def update_weights(self, weights):
        self._params.clear()
        for layer, w in zip(self.layers, weights):
            layer.update_weights(w)
            self._params.append(layer.params)
