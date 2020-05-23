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

def get_r_net(rng_r, num_units):
    layers = []
    layers.append(Dense_no_bias(1))
    layers.append(Reshape((-1)))

    r_network_init, r_network = stax.serial(*layers)
    _, r_network_params = r_network_init(rng_r, (-1, 2 * num_units))

    return r_network, r_network_params

def get_action_r_net(rng_r, num_units):
    r_network_init, r_network = Dense_no_bias(4)
    _, r_network_params = r_network_init(rng_r, (-1, 2 * num_units))

    return r_network, r_network_params

def get_mult_r_net(rng_r, num_units):
    r_network_init, r_network = MultReward(num_units)
    _, r_network_params = r_network_init(rng_r, (-1, num_units, num_units))

    return r_network, r_network_params


def get_mult_c_net(rng_c, num_units):
    c_network_init, c_network = MultRewardUpdate(num_units)
    _, c_network_params = c_network_init(rng_c, (-1, num_units, num_units))
    return c_network, c_network_params

def get_c_net(rng_c, num_units):
    c_network_init, c_network = RewardUpdate(num_units)
    _, c_network_params = c_network_init(rng_c, (-1, num_units))
    return c_network, c_network_params