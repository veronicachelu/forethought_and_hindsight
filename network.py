from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax.numpy as jnp

def Reshape(newshape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (newshape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
  return init_fun, apply_fun

def get_q_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple):

    layers = [stax.Flatten]
    for _ in range(num_hidden_layers):
        layers.append(stax.Dense(num_units))
        layers.append(stax.Relu)
    layers.append(stax.Dense(nA))

    q_network_init, q_network = stax.serial(*layers)

    _, q_network_params = q_network_init(rng, (-1,) + input_dim)

    return q_network, q_network_params

def get_model_network(num_hidden_layers: int,
                  num_units: int,
                  nA: int,
                  rng: List,
                  input_dim: Tuple):
    layers = [stax.Flatten]
    for _ in range(num_hidden_layers):
        layers.append(stax.Dense(num_units))
        layers.append(stax.Relu)
    layers.append(stax.Dense((np.prod(input_dim) + 3) * nA))
    layers.append(Reshape((-1, nA, (np.prod(input_dim) + 3))))

    model_network_init, model_network = stax.serial(*layers)

    _, model_network_params = model_network_init(rng, (-1,) + input_dim)

    return model_network, model_network_params
