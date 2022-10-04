import enum
from typing import Union
from jax import numpy as jnp
import jax
from jax import jit, grad
from jax import lax


class EmbeddingLookup(enum.Enum):
    INDEX = "index"
    ONEHOT = "onehot"


def dense(inputs, weights, bias=None):
    if bias:
        return jnp.add(jnp.dot(inputs, weights), bias)
    else:
        return jnp.dot(inputs, weights)


def conv2d(inputs, weights, strides, padding, dimension_numbers, bias=None):
    output = lax.conv_general_dilated(
        lhs=inputs,
        rhs=weights,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
    )

    if bias:
        output = jnp.add(output, bias)
    return output


# TODO:
def lstm(inputs, lstm_state, weights):
    cell, hidden = lstm_state
    cat_x_h = jnp.concatenate((inputs, hidden), axis=-1)
    forget_gate, input_gate, cell_gate, output_gate = weights
    forget_gate_out = jax.nn.sigmoid(dense(cat_x_h, forget_gate))
    input_gate_out = jax.nn.sigmoid(dense(inputs, input_gate))
    cell_gate_out = jax.nn.tanh(dense(inputs, cell_gate))

    forget_gate_mul_cell = forget_gate_out * cell
    input_mul_cell = input_gate_out * cell_gate_out
    new_context = input_mul_cell + forget_gate_mul_cell

    output_gate_out = jax.nn.sigmoid(dense(inputs, output_gate))
    input_mul_cell_tanh = jax.nn.tanh(input_mul_cell)

    new_hidden_state = output_gate_out * input_mul_cell_tanh

    return new_hidden_state, new_context


# TODO:
def gru(inputs, gru_state, weights):
    pass
