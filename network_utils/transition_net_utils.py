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

def get_b_net(rng_b, num_units):
    b_network_init, b_network = Dense_no_bias(num_units)
    _, b_network_params = b_network_init(rng_b, (-1, num_units))
    return b_network, b_network_params

def get_o_net(rng_o, num_units):
    o_network_init, o_network = Dense_no_bias(num_units)
    _, o_network_params = o_network_init(rng_o, (-1, num_units))

    return o_network, o_network_params



def get_a_net(rng_a, num_units):
    layers = []
    # for _ in range(1):
    #     layers.append(stax.Dense(256))
    #     layers.append(stax.Relu)
    layers.append(Dense_no_bias(num_units * num_units))
    layers.append(Reshape((-1, num_units, num_units)))
    # a_network_init, a_network = FeatureCovariance(num_units)
    a_network_init, a_network = stax.serial(*layers)

    # a_network_init, a_network = Dense_no_bias(num_units*num_units)
    _, a_network_params = a_network_init(rng_a, (-1, num_units))

    return a_network, a_network_params
