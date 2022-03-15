from collections import namedtuple

from jax import numpy as jnp, vmap
from jax.lax import scan
from jax.nn import sigmoid, tanh
from jax.random import normal, split, PRNGKey

LSTM_WEIGHTS = namedtuple('LSTM_WEIGHTS', ('w_ii', 'w_if', 'w_ig', 'w_io', 'w_hi', 'w_hf', 'w_hg', 'w_ho',
                                           'bi', 'bf', 'bg', 'bo'))


def _lstm_cell(state, weights: LSTM_WEIGHTS, input):
    h, c = state
    i = sigmoid(jnp.matmul(input, weights.w_ii) + jnp.matmul(h, weights.w_hi) + weights.bi)
    f = sigmoid(jnp.matmul(input, weights.w_if) + jnp.matmul(h, weights.w_hf) + weights.bf)
    o = sigmoid(jnp.matmul(input, weights.w_io) + jnp.matmul(h, weights.w_ho) + weights.bo)
    g = tanh(jnp.matmul(input, weights.w_ig) + jnp.matmul(c, weights.w_hg) + weights.bg)
    c = f * c + i * g
    h = o * tanh(c)
    return jnp.stack((h, c)), h


def _init_lstm_weights(rng_key, in_dim, hid_dim):
    in_key, hid_key = split(rng_key)
    in_weights = normal(in_key, (4, in_dim, hid_dim))
    hid_weights = normal(in_key, (4, hid_dim, hid_dim))
    bias = jnp.zeros((4, hid_dim))
    return LSTM_WEIGHTS(*in_weights, *hid_weights, *bias)


def rnn(hid_dim=5, num_layers=2):
    def init(rng_seed, in_dim):
        weight_key, state_key = split(rng_seed)
        keys = split(weight_key, num_layers)
        weights = [_init_lstm_weights(keys[0], in_dim, hid_dim)] + [_init_lstm_weights(keys[i], hid_dim, hid_dim) for i
                                                                    in range(1, num_layers)]
        # Note: init_state[:, 0] = hs, init_state[:, 1] = cs
        init_state = normal(state_key, (num_layers, 2, hid_dim))
        return init_state, weights

    def _cell(carry, x):
        states, weights = carry
        out_state = []
        h = x
        for i in range(num_layers):
            new_state, h = _lstm_cell(states[i], weights[i], h)
            out_state.append(new_state)

        return (jnp.stack(out_state), weights), h

    def run_vmap(xs, init_state, weights):
        return vmap(lambda x: scan(_cell, (init_state, weights), x), in_axes=1, out_axes=1)(xs)

    return init, run_vmap


if __name__ == '__main__':
    rng_seed = PRNGKey(43)
    hid_dim = 5
    in_dim = 2
    num_layers = 3
    lengths = 4
    num_datum = 6
    data_seed, init_seed = split(rng_seed)

    x = normal(data_seed, (lengths, num_datum, in_dim))  # time-major

    init, run = rnn(hid_dim=hid_dim, num_layers=num_layers)

    res = run(x, *init(rng_seed=rng_seed, in_dim=in_dim))
