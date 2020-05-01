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
from network_utils import *

def get_pi_net(rng_pi, num_units, nA):
    pi_network_init, pi_network = stax.Dense(nA)
    _, pi_network_params = pi_network_init(rng_pi, (-1, num_units))

    return pi_network, pi_network_params

