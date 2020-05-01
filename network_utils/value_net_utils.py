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

def get_value_net(rng_v, num_units, bias=False):
    if bias == False:
        v_network_init, v_network = Dense_no_bias(1)
    else:
        v_network_init, v_network = stax.Dense(1)
    _, v_network_params = v_network_init(rng_v, (-1, num_units))

    return v_network, v_network_params