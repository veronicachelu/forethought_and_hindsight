from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax.numpy as jnp
from jax import random as jrandom
import haiku as hk

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
                  # double_input_reward_model=Fal
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
    rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_d = jrandom.split(rng, 6)

    if target_networks:
        rng_t, rng_p = jrandom.split(rng_target, 2)
        rng_target_v, rng_target_h, rng_target_o,\
            rng_target_fw_o, rng_target_r, rng_target_d = jrandom.split(rng_t, 6)
        rng_planning_v, rng_planning_h, rng_planning_o, \
        rng_planning_fw_o, rng_planning_r, rng_planning_d = jrandom.split(rng_p, 6)

    h_network, h_network_params = get_h_net(rng_h, num_units, num_hidden_layers, input_size)
    v_network, v_network_params = get_value_net(rng_v, num_units)
    o_network, o_network_params = get_o_net(rng_o, num_units)
    fw_o_network, fw_o_network_params = get_o_net(rng_o, num_units)
    r_network, r_network_params = get_r_net(rng_r, num_units)

    if target_networks:
        target_h_network, target_h_network_params = get_h_net(rng_target_h, num_units, num_hidden_layers, input_size)
        target_v_network, target_v_network_params = get_value_net(rng_target_v, num_units)
        target_o_network, target_o_network_params = get_o_net(rng_target_o, num_units)
        target_fw_o_network, target_fw_o_network_params = get_o_net(rng_target_fw_o, num_units)
        target_r_network, target_r_network_params = get_r_net(rng_target_r, num_units)
        network["target_model"] = {"net": [target_h_network, target_o_network, target_fw_o_network,
                                           target_r_network],
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
    network["model"] = {"net": [h_network, o_network, fw_o_network, r_network],
                        "params": [h_network_params, o_network_params,
                                   fw_o_network_params, r_network_params]
                        }

    return network

# def get_pg_network(num_hidden_layers: int,
#                   num_units: int,
#                   nA: int,
#                   rng: List,
#                   input_dim: Tuple,
#                   target_networks=False,
#                   latent=False,
#                           ):
#     input_size = np.prod(input_dim)
#     num_units = num_units if latent else input_size
#     network = {}
#     rng_pi, rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_d = jrandom.split(rng, 7)
#
#     h_network, h_network_params = get_h_net(rng_h, num_units, input_size)
#     pi_network, pi_network_params = get_pi_net(rng_pi, num_units, nA)
#     v_network, v_network_params = get_value_net(rng_v, num_units)
#     o_network, o_network_params = get_o_net(rng_o, num_units)
#     fw_o_network, fw_o_network_params = get_o_net(rng_o, num_units)
#     r_network, r_network_params = get_r_net(rng_r, num_units)
#     d_network, d_network_params = get_d_net(rng_d, num_units)
#
#     network["value"] = {"net": v_network,
#                         "params": v_network_params}
#     network["pi"] = {"net": pi_network,
#                         "params": pi_network_params}
#     network["model"] = {"net": [h_network, o_network, fw_o_network, r_network, d_network],
#                         "params": [h_network_params, o_network_params,
#                                    fw_o_network_params, r_network_params, d_network_params]
#                         }
#
#     return network

def get_h_net(rng_h, num_units, num_hidden_layers, input_size):
    def network(inputs: jnp.ndarray) -> jnp.ndarray:
        value_output = hk.Linear(num_units, with_bias=False)(inputs)[0]
        return value_output
    net = hk.transform(network)
    dummy_observation = np.zeros((1, input_size), jnp.float32)
    initial_params = net.init(rng_h, dummy_observation)
    return net, initial_params

def get_value_net(rng_v, num_units):
    def network(inputs: jnp.ndarray) -> jnp.ndarray:
        value_output = hk.Linear(1, with_bias=False)(inputs)[0]
        return value_output
    net = hk.transform(network)
    dummy_observation = np.zeros((1, num_units), jnp.float32)
    initial_params = net.init(rng_v, dummy_observation)
    return net, initial_params

def get_o_net(rng_o, num_units):
    def network(inputs: jnp.ndarray) -> jnp.ndarray:
        transition_model_output = hk.Linear(num_units, with_bias=False)(inputs)
        return transition_model_output

    net = hk.transform(network)
    dummy_observation = np.zeros((1, num_units), jnp.float32)
    initial_params = net.init(rng_o, dummy_observation)
    return net, initial_params

def get_r_net(rng_r, num_units):
    def network(inputs: jnp.ndarray) -> jnp.ndarray:
        reward_model_output = hk.Linear(1, with_bias=False)(inputs)[0]
        return reward_model_output

    net = hk.transform(network)
    dummy_observation = np.zeros((1, 2*num_units), jnp.float32)
    initial_params = net.init(rng_r, dummy_observation)
    return net, initial_params
