from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom

def Reshape(newshape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (newshape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
  return init_fun, apply_fun

def get_prediction_model_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  model_class="tabular",
                  double_input_reward_model=False):
    if model_class == "tabular":
        if double_input_reward_model:
            #transitions # fw transitions # actions # rewards # discounts
            return [np.zeros(shape=input_dim + (np.prod(input_dim),)),
             np.zeros(shape=input_dim + (np.prod(input_dim),)),
             np.zeros(shape=input_dim + (np.prod(input_dim),)),
             np.zeros(shape=input_dim + (np.prod(input_dim), nA)), \
             np.zeros(shape=input_dim)], \
            None
        else:
            return [np.zeros(shape=input_dim + (np.prod(input_dim),)),
                    np.zeros(shape=input_dim + (np.prod(input_dim),)),
                np.zeros(shape=input_dim),\
                np.zeros(shape=input_dim)],\
                None
    else:
        # layers = [stax.Flatten]
        # observation
        rng_o, rng_r, rng_d = jrandom.split(rng, 3)

        layers = []
        for _ in range(num_hidden_layers):
            layers.append(stax.Dense(num_units))
            layers.append(stax.Relu)
        layers.append(stax.Dense((np.prod(input_dim))))

        o_network_init, o_network = stax.serial(*layers)
        _, o_network_params = o_network_init(rng_o, (-1,) + input_dim)

        # reward
        layers = []
        for _ in range(num_hidden_layers):
            # layers.append(stax.FanInConcat())
            layers.append(stax.Dense(num_units))
            layers.append(stax.Relu)
        layers.append(stax.Dense(1))
        layers.append(Reshape((-1)))

        r_network_init, r_network = stax.serial(*layers)
        # _, r_network_params = r_network_init(rng_r, (-1,) + (2 * np.prod(input_dim),))
        _, r_network_params = r_network_init(rng_r, (-1,) + (np.prod(input_dim),))

        # gamma/discount/done
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(stax.Dense(num_units))
            layers.append(stax.Relu)
        layers.append(stax.Dense((1)))
        layers.append(Reshape((-1)))

        d_network_init, d_network = stax.serial(*layers)
        _, d_network_params = d_network_init(rng_d, (-1,) + input_dim)

        return [o_network, r_network, d_network],\
               [o_network_params, r_network_params, d_network_params]

def get_prediction_q_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  model_class="tabular"):
    if model_class == "tabular":
        return np.zeros(shape=input_dim), None
    else:
        # layers = [stax.Flatten]
        layers = []
        for _ in range(num_hidden_layers):
            layers.append(stax.Dense(num_units))
            layers.append(stax.Relu)
        layers.append(stax.Dense(1))
        layers.append(Reshape((-1)))

        v_network_init, v_network = stax.serial(*layers)

        _, v_network_params = v_network_init(rng, (-1,) + input_dim)

        return v_network, v_network_params        # layers = [stax.Flatten]


