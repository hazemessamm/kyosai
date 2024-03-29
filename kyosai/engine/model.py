import inspect
import operator as op
from collections import OrderedDict
from typing import Dict, List, NamedTuple, Tuple, Union

import jax
from jax.numpy import DeviceArray
from kyosai import backend, losses, optimizers
from kyosai.engine import data_adapter, generic_utils, graph_recorder
from kyosai.engine.utils import ProgressBar
from kyosai.layers.base_layer import Layer
from kyosai.optimizers.optimizers import Optimizer


class FullPassOutput(NamedTuple):
    predictions: DeviceArray
    loss: DeviceArray
    gradients: Tuple


# Not efficient approach, will be removed or modified
class InnerGraph:
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = generic_utils.flatten(
            inputs
        ), generic_utils.flatten(outputs)
        self._layers_mapping: Dict[str, Layer] = OrderedDict()
        self._dependencies = OrderedDict()
        self._output_names = [output.name for output in self.outputs]
        self._create_graph()
        # generic_utils.jit_call(self)

    @property
    def output(self) -> Union[List, Layer]:
        return self.outputs[0] if len(self.outputs) == 1 else self.outputs

    def _create_graph(self) -> None:
        layers = jax.util.toposort(self.outputs)

        num_inputs = 0
        for layer in layers:
            if len(layer.parents) == 0:
                parents = [f"arg:{num_inputs}"]
                num_inputs += 1
            else:
                parents = [parent_layer.name for parent_layer in layer.parents]
            self._dependencies[layer.name] = parents
            self._layers_mapping[layer.name] = layer
            self._layers = layers
            self.built = True

    def call_with_external_weights(self, weights: Tuple, inputs: DeviceArray, **kwargs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = {f"arg:{i}": inputs[i] for i in range(len(inputs))}

        for weight, (layer, parent_layers) in zip(weights, self._dependencies.items()):
            incoming_inputs = op.itemgetter(*parent_layers)(outputs)
            if (
                isinstance(incoming_inputs, tuple)
                and self._layers_mapping[layer]._call_util._requires_unpacking
            ):
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    weight, *incoming_inputs, **kwargs
                )
            else:
                outputs[layer] = self._layers_mapping[layer].call_with_external_weights(
                    weight, incoming_inputs, **kwargs
                )
        return op.itemgetter(*self._output_names)(outputs)

    def call(self, inputs, **kwargs):
        self.call_with_external_weights(self.weights, inputs, **kwargs)

    def __call__(self, inputs, **kwargs):
        return self.call_with_external_weights(self.weights, inputs, **kwargs)


class _Model(Layer):
    def __init__(self, name=None, trainable=False, sequential=False):
        super(_Model, self).__init__(name=name)
        self.name = backend.memoize(self.__class__.__name__ if name is None else name)
        self.sequential = sequential
        self._compiled = False
        self.trainable = trainable
        self.metrics_instances = {}
        self.history = {}
        self.metrics_values = {}
        self.is_subclass = False

    def compute_output_shape(self, input_shape: Union[List, Tuple]):
        if self.sequential:
            return self.layers[-1].shape
        else:
            if len(self.outputs) == 1:
                return self.outputs[0].shape
            else:
                return [o.shape for o in self.outputs]

    @property
    def layers(self) -> List:
        return self._layers

    @property
    def weights(self) -> List:
        return [l.weights for l in self.layers]

    @property
    def trainable_weights(self) -> List:
        return [l.trainable_weights for l in self.layers]

    @property
    def compiled(self) -> bool:
        return self._compiled

    def _record_graph(self) -> None:
        if not self.built:
            args = inspect.getfullargspec(self.call).args[1:]
            dummy_inputs = [graph_recorder.GraphRecorder() for i in range(len(args))]
            self.call(*dummy_inputs)
            self.is_subclass = True
            inputs = [di.input_layers for di in dummy_inputs]
            # Converting to list because it was a set in the `GraphRecorder` class
            outputs = [list(di.output_layers) for di in dummy_inputs]
            self.inner_graph = InnerGraph(inputs=inputs, outputs=outputs)

    def compile(
        self, loss: losses.Loss, optimizer: Optimizer, metrics: List = None
    ) -> None:
        self._record_graph()

        self.compiled_loss = losses.get(loss)
        self.compiled_optimizer = optimizers.get(optimizer)

        if isinstance(self.compiled_loss, type):
            self.compiled_loss = self.compiled_loss()

        if isinstance(self.compiled_optimizer, type):
            self.compiled_optimizer = self.compiled_optimizer()

        self.compiled_optimizer._initialize(self.weights)
        self._compiled = True

    def set_weights(self, weights: Tuple) -> None:
        "Set new weights on every layer"
        for layer, w in zip(self.layers, weights):
            layer.set_weights(w)

    def update_weights(self, updated_weights: Tuple) -> None:
        for layer, w in zip(self.layers, updated_weights):
            layer.update_weights(w)

    def predict(self, inputs: DeviceArray, **kwargs) -> Union[Tuple, List, DeviceArray]:
        kwargs.pop("training", None)
        return self.call(inputs, training=False, **kwargs)

    def predict_with_external_weights(
        self, weights: Tuple, inputs: Union[Tuple, List, DeviceArray], **kwargs
    ) -> Union[Tuple, List, DeviceArray]:
        kwargs.pop("training", None)
        return self.call_with_external_weights(
            weights, inputs, training=False, **kwargs
        )

    def call(self, inputs, **kwargs):
        raise NotImplementedError("`call` should be implemented in a subclass.")

    def call_with_external_weights(self, weights, inputs, **kwargs):
        raise NotImplementedError(
            "`call_with_external_weights` should be implemented in a subclass."
        )

    def train_step(
        self, x: Union[Tuple, List, DeviceArray], y: Union[Tuple, List, DeviceArray]
    ) -> Dict:
        "Returns loss value and takes training batch"
        forward_backward_output = self.compute_forward_and_backward_pass(x, y)
        self.minimize(self.weights, forward_backward_output.gradients)
        return {"Loss": forward_backward_output.loss}

    def _compute_loss(self, y: DeviceArray, y_pred: DeviceArray) -> DeviceArray:
        return self.compiled_loss(y, y_pred)

    def minimize(self, weights: Tuple, gradients: Tuple):
        new_weights = self.compiled_optimizer.minimize(weights, gradients)
        self.update_weights(new_weights)

    def compute_forward_and_backward_pass(
        self, x: jax.numpy.DeviceArray, y: jax.numpy.DeviceArray
    ) -> FullPassOutput:
        if not self.is_subclass:

            def grad_fn(weights, x, y):
                preds = self.call_with_external_weights(weights, x)
                loss_val = self._compute_loss(y, preds)
                return (loss_val, preds)

        else:

            def grad_fn(weights, x, y):
                preds = self.inner_graph.call_with_external_weights(weights, x)
                loss_val = self._compute_loss(y, preds)
                return (loss_val, preds)

        (loss, predictions), grads = jax.value_and_grad(
            grad_fn, argnums=0, has_aux=True
        )(self.weights, x, y)
        return FullPassOutput(predictions=predictions, loss=loss, gradients=grads)

    def test_step(self, validation_dataset: data_adapter.DataAdapter) -> Dict:
        avg_valid_loss = 0
        for _ in range(validation_dataset.num_batches):
            batch_x, batch_y = validation_dataset.get_batch()
            avg_valid_loss += self.loss(batch_x, batch_y)
        return {"Validation loss": avg_valid_loss}

    def _test_step(self, validation_dataset: data_adapter.DataAdapter) -> Dict:
        if validation_dataset is None:
            return {}
        else:
            return self.test_step(validation_dataset)

    def _assert_compiled(self) -> None:
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

    @property
    def __name__(self):
        return self.name
