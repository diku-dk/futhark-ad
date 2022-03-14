from collections import namedtuple

from jax import numpy as jnp
from jax.lax import scan
from jax.nn import sigmoid, tanh

LSTM_WEIGHTS = namedtuple('LSTM_WEIGHTS', ('w_ii', 'w_hi', 'w_if', 'w_hf', 'w_ig', 'w_hg', 'w_io', 'w_ho',
                                           'bi', 'bf', 'bg', 'bo'))


def lstm_cell(input, state, weights: LSTM_WEIGHTS):
    h, c = state
    i = sigmoid(jnp.matmul(weights.w_ii, input) + jnp.matmul(weights.w_hi @ state + weights.bi))
    f = sigmoid(jnp.matmul(weights.w_if, input) + jnp.matmul(weights.w_hf @ state + weights.bf))
    o = sigmoid(jnp.matmul(weights.w_io, input) + jnp.matmul(weights.w_ho @ state + weights.bo))
    g = tanh(jnp.matmul(weights.w_ig, input) + jnp.matmul(weights.w_hg @ state + weights.bg))
    c = f * c + i * g
    h = o * tanh(c)
    return (h, c)


def rnn(hid_dim=5, num_layers=2, activation=lambda x: x):
    def init():
        weights = []
        return None

    def _cell():
        pass

    def forward(x, init_state, ):
        scan(_cell, init_state)

    return init, forward
