from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax import random as jrandom
from jax.nn.initializers import glorot_normal, normal, ones, zeros
import jax.numpy as jnp
from custom_layers import *

def get_b_net(rng_b, input_size, output_size):
    b_network_init, b_network = Dense_no_bias(output_size)
    _, b_network_params = b_network_init(rng_b, (-1, input_size))
    return b_network, b_network_params

def get_action_b_net(rng_b, input_size, output_size):
    layers = []
    layers.append(Dense_no_bias(output_size * 4))
    layers.append(Reshape((-1, 4, output_size)))
    b_network_init, b_network = stax.serial(*layers)
    _, b_network_params = b_network_init(rng_b, (-1, input_size))
    return b_network, b_network_params

def get_o_net(rng_o, input_size, output_size):
    o_network_init, o_network = Dense_no_bias(output_size)
    _, o_network_params = o_network_init(rng_o, (-1, input_size))

    return o_network, o_network_params

def get_action_o_net(rng_o, input_size, output_size):
    layers = []
    layers.append(Dense_no_bias(output_size * 4))
    layers.append(Reshape((-1, 4, output_size)))
    o_network_init, o_network = stax.serial(*layers)
    _, o_network_params = o_network_init(rng_o, (-1, input_size))
    return o_network, o_network_params

def get_a_net(rng_a, input_size, output_size):
    layers = []
    layers.append(Dense_no_bias(output_size * output_size))
    layers.append(Reshape((-1, output_size, output_size)))
    a_network_init, a_network = stax.serial(*layers)

    _, a_network_params = a_network_init(rng_a, (-1, input_size))

    return a_network, a_network_params

def get_action_a_net(rng_a, input_size, output_size):
    layers = []
    layers.append(Dense_no_bias(output_size * output_size * 4))
    layers.append(Reshape((-1, 4, output_size, output_size)))
    a_network_init, a_network = stax.serial(*layers)

    _, a_network_params = a_network_init(rng_a, (-1, input_size))

    return a_network, a_network_params
