from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
from jax.nn.initializers import glorot_normal, normal, ones, zeros
import jax.numpy as jnp

def Reshape(newshape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (newshape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
  return init_fun, apply_fun

def get_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  rng_target: List,
                  input_dim: Tuple,
                  model_class="tabular",
                  model_family="extrinsic",
                  target_networks=False,
                  pg=False,
                  latent=False,
                  feature_coder=None,
                  # double_input_reward_model=False
                ):
    if feature_coder is not None:
        input_dim = get_input_dim(input_dim, feature_coder)
    if pg:
        return get_pg_network(num_hidden_layers, num_units, nA,
                            rng, input_dim, latent)
    if model_class == "tabular":
        return get_tabular_network(num_hidden_layers, num_units, nA,
                            rng, input_dim)
    if model_family == "extrinsic":
        return get_extrinsic_network(num_hidden_layers, num_units, nA,
                        rng, input_dim)
    if model_family == "q":
        return get_q_network(num_hidden_layers, num_units, nA,
                        rng, input_dim)
    return get_intrinsic_network(num_hidden_layers, num_units, nA,
                rng, rng_target, input_dim, target_networks, latent)

def get_input_dim(input_dim, feature_coder):
    if feature_coder["type"] == "tile":
        return np.prod(feature_coder["num_tiles"]) * feature_coder["num_tilings"]
    else:
        return input_dim

def get_tabular_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                        ):

    network = {}
    network["value"] = {"net": np.zeros(shape=input_dim),
                        "params": None
                        }
    network["model"] = {"net": [np.zeros(shape=input_dim + (np.prod(input_dim),)),
                                np.zeros(shape=input_dim + (np.prod(input_dim),)),
                                np.zeros(shape=input_dim + (np.prod(input_dim),)), \
                                # np.zeros(shape=input_dim + (2,))],\
                                np.zeros(shape=input_dim)], \
                        "params": None
                        }
    return network

def get_extrinsic_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                          ):

    input_size = np.prod(input_dim)
    network = {}
    rng_v, rng_h, rng_o, rng_fw_o, rng_r = jrandom.split(rng, 5)

    h_network, h_network_params = get_h_net(rng_h, num_units, num_hidden_layers, input_size)
    v_network, v_network_params = get_value_net(rng_v, input_size)
    o_network, o_network_params = get_o_net(rng_o, input_size)
    fw_o_network, fw_o_network_params = get_o_net(rng_fw_o, input_size)
    r_network, r_network_params = get_r_net(rng_r, input_size)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["model"] = {"net": [o_network, fw_o_network, r_network],
                        "params": [o_network_params,
                                   fw_o_network_params, r_network_params]
                        }

    return network

def get_q_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  # double_input_reward_model=False
                          ):

    input_size = np.prod(input_dim)
    network = {}

    q_network_init, q_network = stax.Dense(nA)
    _, q_network_params = q_network_init(rng, (-1, input_size))
    network["qvalue"] = {"net": q_network,
                        "params": q_network_params}  # layers = [stax.Flatten]

    return network

def get_intrinsic_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  rng_target: List,
                  input_dim: Tuple,
                  target_networks=False,
                  latent=False,
                          ):

    input_size = np.prod(input_dim)
    num_units = num_units if latent else input_size
    network = {}
    rng_v, rng_h, rng_o, rng_fw_o, rng_r = jrandom.split(rng, 5)

    if target_networks:
        rng_t, rng_p = jrandom.split(rng_target, 2)
        rng_target_v, rng_target_h, rng_target_o,\
            rng_target_fw_o, rng_target_r, rng_target_d = jrandom.split(rng_t, 6)
        rng_planning_v, rng_planning_h, rng_planning_o, \
        rng_planning_fw_o, rng_planning_r, rng_planning_d = jrandom.split(rng_p, 6)

    h_network, h_network_params = get_h_net(rng_h, num_units, num_hidden_layers, input_size)
    v_network, v_network_params = get_value_net(rng_v, num_units)
    o_network, o_network_params = get_o_net(rng_o, num_units)
    fw_o_network, fw_o_network_params = get_o_net(rng_fw_o, num_units)
    r_network, r_network_params = get_r_net(rng_r, num_units)

    if target_networks:
        target_h_network, target_h_network_params = get_h_net(rng_target_h, num_units, num_hidden_layers, input_size)
        target_v_network, target_v_network_params = get_value_net(rng_target_v, num_units)
        target_o_network, target_o_network_params = get_o_net(rng_target_o, num_units)
        target_fw_o_network, target_fw_o_network_params = get_o_net(rng_target_fw_o, num_units)
        target_r_network, target_r_network_params = get_r_net(rng_target_r, num_units)
        network["target_model"] = {"net": [target_h_network, target_o_network, target_fw_o_network,
                                           target_r_network, ],
                            "params": [target_h_network_params, target_o_network_params,
                                       target_fw_o_network_params, target_r_network_params,
                                       ]
                            }
        network["target_value"] = {"net": target_v_network,
                            "params": target_v_network_params}

        planning_v_network, planning_v_network_params = get_value_net(rng_planning_v, num_units)
        network["planning_value"] = {"net": planning_v_network,
                                   "params": planning_v_network_params}

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["model"] = {"net": [h_network, o_network, fw_o_network, r_network, ],
                        "params": [h_network_params, o_network_params,
                                   fw_o_network_params, r_network_params, ]
                        }

    return network

def get_pg_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                  target_networks=False,
                  latent=False,
                          ):
    input_size = np.prod(input_dim)
    num_units = num_units if latent else input_size
    network = {}
    rng_pi, rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_d = jrandom.split(rng, 7)

    h_network, h_network_params = get_h_net(rng_h, num_units, input_size)
    pi_network, pi_network_params = get_pi_net(rng_pi, num_units, nA)
    v_network, v_network_params = get_value_net(rng_v, num_units)
    o_network, o_network_params = get_o_net(rng_o, num_units)
    fw_o_network, fw_o_network_params = get_o_net(rng_o, num_units)
    r_network, r_network_params = get_r_net(rng_r, num_units)
    d_network, d_network_params = get_d_net(rng_d, num_units)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["pi"] = {"net": pi_network,
                        "params": pi_network_params}
    network["model"] = {"net": [h_network, o_network, fw_o_network, r_network, d_network],
                        "params": [h_network_params, o_network_params,
                                   fw_o_network_params, r_network_params, d_network_params]
                        }

    return network

def Dense_no_bias(out_dim, W_init=glorot_normal()):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = jrandom.split(rng)
    W = W_init(k1, (input_shape[-1], out_dim))
    return output_shape, (W)
  def apply_fun(params, inputs, **kwargs):
    W = params
    return jnp.dot(inputs, W)
  return init_fun, apply_fun

def get_h_net(rng_h, num_units, num_hidden_layers, input_size):
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(Dense_no_bias(num_units))
        layers.append(stax.Relu)
    layers.append(Dense_no_bias(num_units))
    # h_network_init, h_network = stax.Dense(num_units)
    h_network_init, h_network = stax.serial(*layers)
    _, h_network_params = h_network_init(rng_h, (-1, input_size))
    return h_network, h_network_params

def get_pi_net(rng_pi, num_units, nA):
    pi_network_init, pi_network = Dense_no_bias(nA)
    _, pi_network_params = pi_network_init(rng_pi, (-1, num_units))

    return pi_network, pi_network_params

def get_value_net(rng_v, num_units):
    layers = []
    layers.append(Dense_no_bias(1))
    layers.append(Reshape((-1)))
    v_network_init, v_network = stax.serial(*layers)
    _, v_network_params = v_network_init(rng_v, (-1, num_units))

    return v_network, v_network_params

def get_o_net(rng_o, num_units):
    o_network_init, o_network = Dense_no_bias(num_units)
    _, o_network_params = o_network_init(rng_o, (-1, num_units))

    return o_network, o_network_params

def get_r_net(rng_r, num_units):
    layers = []
    layers.append(Dense_no_bias(1))
    layers.append(Reshape((-1)))

    r_network_init, r_network = stax.serial(*layers)
    # if double_input_reward_model:
    _, r_network_params = r_network_init(rng_r, (-1, 2 * num_units))
    # else:
    #     _, r_network_params = r_network_init(rng_r, (-1, num_units))

    return r_network, r_network_params

def get_d_net(rng_d, num_units):
    layers = []
    layers.append(Dense_no_bias(2))
    layers.append(Reshape((-1)))

    d_network_init, d_network = stax.serial(*layers)
    _, d_network_params = d_network_init(rng_d, (-1, 2 * num_units))

    return d_network, d_network_params