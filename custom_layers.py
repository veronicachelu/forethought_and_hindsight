from jax.experimental import stax
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from jax import random as jrandom
from jax.nn.initializers import glorot_normal, normal, ones, zeros
import jax.numpy as jnp

def Reshape(newshape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (newshape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, newshape)
  return init_fun, apply_fun

def Dense_no_bias(out_dim, W_init=glorot_normal()):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    W = W_init(rng, (input_shape[-1], out_dim))
    return output_shape, W
  def apply_fun(params, inputs, **kwargs):
    return jnp.dot(inputs, params)
  return init_fun, apply_fun

def MultReward(out_dim, W_init=glorot_normal()):
    def init_fun(rng, feature_shape):
        output_shape = feature_shape[:-1] + (1,)
        W = W_init(rng, (feature_shape[-1], feature_shape[-1]))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        predecessor, current = inputs
        second_product = lax.batch_matmul(params[None, ...], current[..., None])
        first_product = lax.batch_matmul(jnp.transpose(predecessor[..., None],
                                         axes=[0, 2, 1]),
                                         second_product)
        return jnp.squeeze(first_product, axis=[-1, -2])

    return init_fun, apply_fun

def RewardUpdate(out_dim, W_init=glorot_normal()):
    def init_fun(rng, feature_shape):
        # output_shape = feature_shape[:-1] + (out_dim,)
        # W = W_init(rng, (feature_shape[-1], out_dim))
        output_shape = feature_shape[:-1] + (feature_shape[-1],)
        W = W_init(rng, (feature_shape[-1] * 2, 1))
        # W = W_init(rng, (feature_shape[-1], out_dim))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        # expected_predecessors, current = inputs
        # input = jnp.concatenate([expected_predecessors, current], axis=-1)
        # output = jnp.dot(input, params)
        # return output
        cross_predecessors, expected_predecessors, current = inputs
        b = cross_predecessors.shape[0]
        params_shape = params.shape[0]
        first_half = jnp.tile(params[:params_shape//2, :][None, ...], (b, 1, 1))
        pred_contrib = lax.batch_matmul(cross_predecessors,
                                        first_half)

        second_half = jnp.tile(params[params_shape//2:, :][None, ...], (b, 1, 1))
        cross_exp_pred_current = lax.batch_matmul(expected_predecessors[..., None],
                         jnp.transpose(current[..., None], axes=[0, 2, 1]))

        current_contrib = lax.batch_matmul(cross_exp_pred_current, second_half)
        # first_product = jnp.squeeze(lax.batch_matmul(expected_predecessors[..., None],
        #                                              second_product), axis=-1)
        return jnp.squeeze(pred_contrib + current_contrib, axis=[-1])
        # cross_predecessors, current = inputs
        # second_product = lax.batch_matmul(params[None, ...], current[..., None])
        # first_product = lax.batch_matmul(cross_predecessors, second_product)
        # return jnp.squeeze(first_product, axis=-1)

    return init_fun, apply_fun


def MultRewardUpdate(out_dim, W_init=glorot_normal()):
    def init_fun(rng, feature_shape):
        output_shape = feature_shape[:-1] + (feature_shape[-1],)
        W = W_init(rng, (feature_shape[-1], feature_shape[-1]))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        cross_predecessors, current = inputs
        second_product = lax.batch_matmul(params[None, ...], current[..., None])
        first_product = lax.batch_matmul(cross_predecessors, second_product)
        return jnp.squeeze(first_product, axis=-1)

    return init_fun, apply_fun

def UpdateLayer(out_dim, W_init=glorot_normal()):
    def init_fun(rng, feature_shape):
        output_shape = feature_shape[:-1] + (feature_shape[-1],)
        W = W_init(rng, (feature_shape[-1], feature_shape[-1]))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        current, v_params = inputs
        second_product = lax.batch_matmul(params[None, ...], v_params[None, ...])
        first_product = lax.batch_matmul(jnp.transpose(current[..., None],
                                                       axes=[0, 2, 1]),
                                         second_product)
        return jnp.squeeze(first_product, axis=-1)

    return init_fun, apply_fun