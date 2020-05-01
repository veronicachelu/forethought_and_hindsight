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
                ):
    if feature_coder is not None:
        input_dim = get_input_dim(input_dim, feature_coder)
    if model_family == "ac":
        return get_pg_network(num_hidden_layers, num_units, nA,
                            rng, input_dim, latent)
    if model_class == "tabular":
        return get_tabular_network(num_hidden_layers, num_units, nA,
                            rng, input_dim)

    if model_family == "extrinsic":
        return get_extrinsic_network(num_hidden_layers, num_units, nA,
                        rng, input_dim)

    if model_family == "mult_extrinsic":
        return get_mult_extrinsic_network(num_hidden_layers, num_units, nA,
                        rng, input_dim)

    if model_family == "true":
        return get_true_network(num_hidden_layers, num_units, nA,
                                rng, input_dim)
    if model_family == "true_mult":
        return get_mult_true_network(num_hidden_layers, num_units, nA,
                                rng, input_dim)

    if model_family == "q":
        return get_q_network(num_hidden_layers, num_units, nA,
                        rng, input_dim)

    if model_family == "intrinsic":
        return get_intrinsic_network(num_hidden_layers, num_units, nA,
                rng, rng_target, input_dim, target_networks, latent)

    if model_family == "mult_intrinsic":
        return get_mult_intrinsic_network(num_hidden_layers, num_units, nA,
                rng, rng_target, input_dim, target_networks, latent)

    if model_family == "update":
        return get_mult_update_network(num_hidden_layers, num_units, nA,
                rng, rng_target, input_dim, target_networks, latent)


def get_input_dim(input_dim, feature_coder):
    if feature_coder["type"] == "tile":
        return np.prod(feature_coder["num_tiles"]) * feature_coder["num_tilings"]
    elif feature_coder["type"] == "rbf":
        return np.prod(feature_coder["num_centers"])
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

def get_true_network(num_hidden_layers: int,
                     num_units: int,
                     nA: int,
                     rng: List,
                     input_dim: Tuple,
                     ):
    input_size = np.prod(input_dim)
    network = {}
    rng_v, _, rng_b, rng_a, rng_c = jrandom.split(rng, 5)

    v_network, v_network_params = get_value_net(rng_v, input_size, bias=False)
    c_network, c_network_params = get_c_net(rng_c, input_size)
    a_network, a_network_params = get_a_net(rng_a, input_size)
    b_network, b_network_params = get_b_net(rng_b, input_size)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["model"] = {"net": [b_network, a_network, c_network],
                        "params": [b_network_params,
                                   a_network_params,
                                   c_network_params]
                        }

    return network

def get_mult_true_network(num_hidden_layers: int,
                     num_units: int,
                     nA: int,
                     rng: List,
                     input_dim: Tuple,
                     ):
    input_size = np.prod(input_dim)
    network = {}
    rng_v, _, rng_b, rng_a, rng_c = jrandom.split(rng, 5)

    v_network, v_network_params = get_value_net(rng_v, input_size, bias=False)
    c_network, c_network_params = get_mult_c_net(rng_c, input_size)
    a_network, a_network_params = get_a_net(rng_a, input_size)
    b_network, b_network_params = get_b_net(rng_b, input_size)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["model"] = {"net": [b_network, a_network, c_network],
                        "params": [b_network_params,
                                   a_network_params,
                                   c_network_params]
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
    rng_v, _, rng_o, rng_fw_o, rng_r = jrandom.split(rng, 5)

    v_network, v_network_params = get_value_net(rng_v, input_size, bias=False)
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

def get_mult_extrinsic_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple,
                          ):
    input_size = np.prod(input_dim)
    network = {}
    rng_v, _, rng_o, rng_fw_o, rng_r = jrandom.split(rng, 5)

    v_network, v_network_params = get_value_net(rng_v, input_size, bias=False)
    o_network, o_network_params = get_o_net(rng_o, input_size)
    fw_o_network, fw_o_network_params = get_o_net(rng_fw_o, input_size)
    r_network, r_network_params = get_mult_r_net(rng_r, input_size)

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
        target_v_network, target_v_network_params = get_value_net(rng_target_v, num_units, bias=False)
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

def get_mult_intrinsic_network(num_hidden_layers: int,
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
    r_network, r_network_params = get_mult_r_net(rng_r, num_units)

    if target_networks:
        target_h_network, target_h_network_params = get_h_net(rng_target_h, num_units, num_hidden_layers, input_size)
        target_v_network, target_v_network_params = get_value_net(rng_target_v, num_units, bias=False)
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

def get_mult_update_network(num_hidden_layers: int,
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
    rng_v, rng_h, rng_updt, _, _ = jrandom.split(rng, 5)

    h_network, h_network_params = get_h_net(rng_h, num_units, 0, input_size)
    v_network, v_network_params = get_value_net(rng_v, num_units)
    updt_network, updt_network_params = get_updt_net(rng_updt, num_units, num_hidden_layers)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["model"] = {"net": [h_network, updt_network],
                        "params": [h_network_params, updt_network_params]
                        }

    return network

def get_updt_net(rng_updt, num_units, num_hidden_layers):
    net_layers = []
    parallel_layers = []
    # for _ in range(2):
    #     layers = []
    #     for _ in range(num_hidden_layers):
    #         layers.append(stax.Dense(num_units))
    #         layers.append(stax.Relu)
    #     layers.append(stax.Dense(num_units))
    #     layers = stax.serial(*layers)
    #     parallel_layers.append(layers)
    # net_layers.append(stax.parallel(*parallel_layers))
    net_layers.append(stax.FanInConcat())
    # net_layers.append(stax.Dense(num_units))
    net_layers.append(Dense_no_bias(num_units))

    updt_network_init, updt_network = stax.serial(*net_layers)

    # updt_network_init, updt_network = UpdateLayer(num_units)
    _, updt_network_params = updt_network_init(rng_updt, [(-1, num_units), (-1, num_units)])
    return updt_network, updt_network_params


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
    rng_v, rng_h, rng_o, rng_fw_o, rng_r, rng_pi = jrandom.split(rng, 6)

    h_network, h_network_params = get_h_net(rng_h, num_units, num_hidden_layers, input_size)
    pi_network, pi_network_params = get_pi_net(rng_pi, num_units, nA)
    v_network, v_network_params = get_value_net(rng_v, num_units, bias=True)
    o_network, o_network_params = get_o_net(rng_o, num_units)
    fw_o_network, fw_o_network_params = get_o_net(rng_o, num_units)
    r_network, r_network_params = get_r_net(rng_r, num_units)

    network["value"] = {"net": v_network,
                        "params": v_network_params}
    network["pi"] = {"net": pi_network,
                        "params": pi_network_params}
    network["model"] = {"net": [h_network, o_network, fw_o_network, r_network],
                        "params": [h_network_params, o_network_params,
                                   fw_o_network_params, r_network_params]
                        }

    return network