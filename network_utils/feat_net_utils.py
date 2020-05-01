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

def get_h_net(rng_h, num_units, num_hidden_layers, input_size):
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(stax.Dense(num_units))
        layers.append(stax.Relu)
    layers.append(stax.Dense(num_units))

    h_network_init, h_network = stax.serial(*layers)
    _, h_network_params = h_network_init(rng_h, (-1, input_size))
    return h_network, h_network_params
