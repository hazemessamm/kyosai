from functools import partial
import inspect
from typing import NamedTuple
from kyosai import backend, losses, optimizers
from kyosai.engine import data_adapter
from kyosai.engine.utils import ProgressBar
from kyosai.layers import base_layer
import jax


class BackwardOutput(NamedTuple):
    predictions: jax.numpy.DeviceArray
    loss: jax.numpy.DeviceArray
    gradients: tuple



class _Model(base_layer.Layer):
    def __init__(self, name=None, trainable=False, sequential=False):
        super(_Model, self).__init__(name=name)
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        self.sequential = sequential
        self._compiled = False
        self.trainable = trainable
        self.metrics_instances = {}
        self._params = []
        self.history = {}
        self.metrics_values = {}

    @property
    def __name__(self):
        return self.name

    @property
    def layers(self):
        return self._layers

    @property
    def params(self):
        if self._params:
            return self._params
        elif self.layers:
            return [layer.params for layer in self.layers if layer.built]
        else:
            return []

    @property
    def weights(self):
        return self.params

    @property
    def compiled(self):
        return self._compiled

    def simulate_call(self):
        if not self.built:
            args = inspect.getfullargspec(self.call).args[1:]
            dummy_inputs = [base_layer.DummyInput() for i in range(len(args))]
            self(*dummy_inputs)

    def compile(self, loss, optimizer, metrics=None):
        self.compiled_loss = losses.get(loss)
        self.compiled_optimizer = optimizers.get(optimizer)

        if isinstance(self.compiled_loss, type):
            self.compiled_loss = self.compiled_loss()

        if isinstance(self.compiled_optimizer, type):
            self.compiled_optimizer = self.compiled_optimizer()

        self.simulate_call()
        self.compiled_optimizer._initialize(self.params)
        self._compiled = True

    # TODO: implement `build()` method
    def build(self, input_shape):
        pass

    def get_weights(self):
        return self._params

    def set_weights(self, weights):
        "Set new weights on every layer"
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)
        self._params = weights

    def update_weights(self, updated_weights):
        for layer, w in zip(self.layers, updated_weights):
            layer.update_weights(w)
        self._params = updated_weights

    def predict(self, inputs, **kwargs):
        kwargs.pop("training", None)
        return self.call(inputs, training=False, **kwargs)

    def predict_with_external_weights(self, params, inputs, **kwargs):
        kwargs.pop("training", None)
        return self.call_with_external_weights(params, inputs, training=False, **kwargs)

    def call(self, inputs, **kwargs):
        raise NotImplementedError("`call` should be implemented in a subclass.")

    def call_with_external_weights(self, params, inputs, **kwargs):
        raise NotImplementedError(
            "`call_with_external_weights` should be implemented in a subclass."
        )

    def train_step(self, x, y):
        "Returns loss value and takes training batch"
        backward_output = self.compute_gradients(x, y)
        params = self.compiled_optimizer.minimize(self.params, backward_output.gradients)
        self.update_weights(params)
        return {"Loss": backward_output.loss}

    def _compute_loss(self, y, y_pred):
        return self.compiled_loss(y, y_pred)

    def compute_gradients(self, x, y) -> BackwardOutput:
        def grad_fn(params, x, y):
            preds = self.call_with_external_weights(params, x)
            loss_val = self._compute_loss(y, preds)
            return (loss_val, preds)
        
        (loss, predictions), grads = jax.value_and_grad(grad_fn, argnums=0, has_aux=True)(self.params, x, y)
        return BackwardOutput(predictions=predictions, loss=loss, gradients=grads)

    def test_step(self, validation_dataset):
        avg_valid_loss = 0
        for _ in range(validation_dataset.num_batches):
            batch_x, batch_y = validation_dataset.get_batch()
            avg_valid_loss += self.loss(batch_x, batch_y)
        return {"Validation loss": avg_valid_loss}

    def _test_step(self, validation_dataset):
        if validation_dataset is None:
            return {}
        else:
            return self.test_step(validation_dataset)

    def _assert_compiled(self):
        if not self.compiled:
            raise Exception("Model is not compiled, use `compile()` method")

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

        self._assert_compiled()
        dataset = data_adapter.TensorLikeDataAdapter(
            x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=shuffle
        )

        if validation_data:
            validation_dataset = data_adapter.TensorLikeDataAdapter(
                x, y, batch_size=batch_size, epochs=epochs, steps=steps, shuffle=False
            )
        else:
            validation_dataset = None

        progbar = ProgressBar(len(dataset))
        # TODO: should add support for logging metrics
        losses = {}
        for epoch in range(1, epochs + 1):
            self.metrics_values.clear()
            for step in range(1, len(dataset) + 1):
                batch_x, batch_y = dataset.get_batch()
                training_losses = self.train_step(batch_x, batch_y)
                validation_losses = self._test_step(validation_dataset)
                losses.update(training_losses)
                losses.update(validation_losses)
                progbar.update(epoch, step, **losses)
            progbar.reset()
