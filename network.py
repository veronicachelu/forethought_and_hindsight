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

def get_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  model_class="tabular",
                  model_family="extrinsic",
                  # double_input_reward_model=False
                ):

    if model_class == "tabular":
        return get_tabular_network(num_hidden_layers, num_units, nA,
                            rng, input_dim)
    else:
        if model_family == "extrinsic":
            return get_extrinsic_network(num_hidden_layers, num_units, nA,
                            rng, input_dim)
        else:
            return get_intrinsic_network(num_hidden_layers, num_units, nA,
                                  rng, input_dim)



def get_tabular_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  # double_input_reward_model=False
                        ):

    network = {}
    network["value"] = {"net": np.zeros(shape=input_dim),
                        "params": None
                        }
    # if double_input_reward_model:
        # transitions # fw transitions # rewards # discounts
    network["model"] = {"net": [np.zeros(shape=input_dim + (np.prod(input_dim),)),
                                np.zeros(shape=input_dim + (np.prod(input_dim),)),
                                np.zeros(shape=input_dim + (np.prod(input_dim),)), \
                                # np.zeros(shape=input_dim + (2,))],\
                                np.zeros(shape=input_dim)], \
                        "params": None
                        }
    # else:
    #     network["model"] = {"net": [np.zeros(shape=input_dim + (np.prod(input_dim),)),
    #                                 np.zeros(shape=input_dim + (np.prod(input_dim),)),
    #                                 np.zeros(shape=input_dim), \
    #                                 # np.zeros(shape=input_dim + (2,))],\
    #                                 np.zeros(shape=input_dim)], \
    #                         "params": None
    #                         }

    return network

def get_extrinsic_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  # double_input_reward_model=False
                          ):

    input_size = np.prod(input_dim)
    network = {}
    rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_d = jrandom.split(rng, 6)

    # layers = [stax.Flatten]
    layers = []
    # for _ in range(num_hidden_layers):
    #     layers.append(stax.Dense(num_units))
    #     layers.append(stax.Relu)
    layers.append(stax.Dense(1))
    layers.append(Reshape((-1)))

    v_network_init, v_network = stax.serial(*layers)

    _, v_network_params = v_network_init(rng_v, (-1, input_size))

    network["value"] = {"net": v_network,
                        "params": v_network_params}  # layers = [stax.Flatten]

    # layers = []
    # for _ in range(num_hidden_layers):
    #     layers.append(stax.Dense(num_units))
    #     layers.append(stax.Relu)
    # layers.append(stax.Dense(input_size))

    o_network_init, o_network = stax.Dense(input_size)
    _, o_network_params = o_network_init(rng_o, (-1, input_size))

    # layers = []
    # for _ in range(num_hidden_layers):
    #     layers.append(stax.Dense(num_units))
    #     layers.append(stax.Relu)
    # layers.append(stax.Dense(input_size))

    fw_o_network_init, fw_o_network = stax.Dense(input_size)
    _, fw_o_network_params = fw_o_network_init(rng_fw_o, (-1, input_size))

    # reward
    layers = []
    # for _ in range(num_hidden_layers):
        # layers.append(stax.FanInConcat())
        # layers.append(stax.Dense(num_units))
        # layers.append(stax.Relu)

    layers.append(stax.Dense(1))
    layers.append(Reshape((-1)))

    r_network_init, r_network = stax.serial(*layers)
    # if double_input_reward_model:
    _, r_network_params = r_network_init(rng_r, (-1, 2 * input_size))
    # else:
    #     _, r_network_params = r_network_init(rng_r, (-1, input_size))

    # gamma/discount/done
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(stax.Dense(num_units))
        layers.append(stax.Relu)
    layers.append(stax.Dense((1)))
    layers.append(Reshape((-1)))

    d_network_init, d_network = stax.serial(*layers)
    _, d_network_params = d_network_init(rng_d, (-1, input_size))

    network["model"] = {"net": [o_network, fw_o_network, r_network, None],
                        "params": [o_network_params, fw_o_network_params,
                                   r_network_params, None]
                        }

    return network

def get_intrinsic_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  # double_input_reward_model=False
                          ):

    intput_size = np.prod(input_dim)
    num_units = intput_size
    network = {}
    rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_d = jrandom.split(rng, 6)

    h_network_init, h_network = stax.Dense(num_units)
    _, h_network_params = h_network_init(rng_h, (-1, intput_size))

    layers = []
    layers.append(stax.Dense(1))
    layers.append(Reshape((-1)))

    v_network_init, v_network = stax.serial(*layers)

    _, v_network_params = v_network_init(rng_v, (-1, num_units))

    network["value"] = {"net": v_network,
                        "params": v_network_params}

    o_network_init, o_network = stax.Dense(num_units)
    _, o_network_params = o_network_init(rng_o, (-1, num_units))

    fw_o_network_init, fw_o_network = stax.Dense(num_units)
    _, fw_o_network_params = fw_o_network_init(rng_fw_o, (-1, num_units))

    layers = []
    layers.append(stax.Dense(1))
    layers.append(Reshape((-1)))

    r_network_init, r_network = stax.serial(*layers)
    # if double_input_reward_model:
    _, r_network_params = r_network_init(rng_r, (-1, 2 * num_units))
    # else:
    #     _, r_network_params = r_network_init(rng_r, (-1, num_units))

    layers = []
    layers.append(stax.Dense(2))
    layers.append(Reshape((-1)))

    d_network_init, d_network = stax.serial(*layers)
    _, d_network_params = d_network_init(rng_d, (-1, 2 * num_units))

    network["model"] = {"net": [h_network, o_network, fw_o_network, r_network, d_network],
                        "params": [h_network_params, o_network_params,
                                   fw_o_network_params, r_network_params, d_network_params]
                        }

    return network