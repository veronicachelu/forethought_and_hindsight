from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import numpy as np
from agents.agent import Agent
import tensorflow as tf
import rlax
from basis.feature_mapper import FeatureMapper
import functools
from control_agents.linear.Q.vanilla_q import VanillaQ
NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TrueBwQ(VanillaQ):
    def __init__(
            self,
            **kwargs
    ):
        super(TrueBwQ, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        def model_loss(a_params,
                       b_params,
                       c_params,
                       transitions):
            o_tmn = transitions[0][0]
            a_tmn = transitions[0][1]
            o_t = transitions[-1][-1]

            cross_o_tmn_target = lax.batch_matmul(o_tmn[..., None],
                                                  jnp.transpose(o_tmn[..., None],
                                                                axes=[0, 2, 1]))

            model_expected_o_tmn = jax.vmap(lambda
                                   model_exp_otmn, a:
                                   model_exp_otmn[a])(self._b_network(b_params, o_t), a_tmn)
            b_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_expected_o_tmn, o_tmn))
            model_cross_o_tmn = jax.vmap(lambda
                                    model_cross_otmn, a:
                                    model_cross_otmn[a])(self._a_network(a_params, o_t), a_tmn)
            a_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_cross_o_tmn, cross_o_tmn_target))

            model_vector_r_tmn = self._c_network(c_params,
                                                 (lax.stop_gradient(model_cross_o_tmn),
                                                  lax.stop_gradient(model_expected_o_tmn),
                                                  o_t))

            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]
            vector_r_t_target = o_tmn * r_t_target[..., None]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_vector_r_tmn, vector_r_t_target))
            total_loss = b_loss + r_loss + a_loss

            return total_loss, {"cross_loss(A)": a_loss,
                               "expected_loss(b)": b_loss,
                               "r_loss(w_r)": r_loss,
                               "r_loss": r_loss,
                               }

        def q_planning_grad(q_params, a_params, b_params, c_params, x, d):
            x_per_a = jnp.tile(jnp.expand_dims(x, 1), (1, self._nA, 1))
            d_per_a = jnp.tile(jnp.expand_dims(jnp.array([(self._discount ** self._n)]) * d, 1), (1, self._nA))
            d_per_a_flat = jnp.reshape(d_per_a, (-1))
            exp_prev_x_per_a = self._b_forward(b_params, x)
            cross_prev_x_per_a = self._a_forward(a_params, x)
            b, na, nf = exp_prev_x_per_a.shape
            exp_prev_x = jnp.reshape(exp_prev_x_per_a, (-1, nf))
            cross_prev_x = jnp.reshape(cross_prev_x_per_a, (-1, nf, nf))
            x_flat = jnp.reshape(x_per_a, (-1, nf))

            vector_r_flat = self._c_network(c_params,
                        (cross_prev_x, exp_prev_x, x_flat))[..., None]
            vector_r_per_a = jnp.reshape(vector_r_flat, (b, na, nf, 1))

            # cross_prev_current_flat = jnp.einsum("bfi, bif->bff", exp_prev_x[..., None],
            #                            jnp.transpose(x_flat[..., None], axes=[0, 2, 1]))
            # q_params_tiled_flat = jnp.tile(q_params[None, ...], (b * self._nA, 1, 1))
            # q_flat = jnp.einsum("bif,bfk->bik", cross_prev_current_flat, q_params_tiled_flat)
            # max_q_flat = d_per_a_flat[..., None] * jnp.array([self._discount ** self._n]) * \
            #     jnp.max(q_flat, axis=-1)
            # max_q_flat_per_a = jnp.reshape(max_q_flat, (b, na, nf, 1))

            max_q = jnp.max(self._q_forward(q_params, x), axis=-1)
            max_q_per_a = jnp.tile(jnp.expand_dims(max_q, 1), (1, self._nA))
            max_q_per_a_flat = jnp.reshape(max_q_per_a, (-1))
            vector_max_q_per_a_flat = d_per_a_flat[..., None] * exp_prev_x * max_q_per_a_flat[..., None]
            vector_max_q_per_a = jnp.reshape(vector_max_q_per_a_flat, (b, na, nf, 1))

            q_params_tiled_flat = jnp.tile(q_params[None, ...], (b * self._nA, 1, 1))
            vector_prev_q_flat = jnp.einsum("bij,bjk->bik", cross_prev_x, q_params_tiled_flat)
            a_per_a = jnp.tile(jnp.expand_dims(jnp.arange(self._nA), 0),
                               (b, 1))
            the_as = jnp.reshape(a_per_a, (-1))
            vector_prev_q = jax.vmap(lambda q, a: q[jnp.arange(nf), a], in_axes=(0, 0), out_axes=(0))(vector_prev_q_flat, the_as)
            vector_prev_q_per_a = jnp.reshape(vector_prev_q, (b, na, nf, 1))

            q_grad = jnp.mean((vector_r_per_a + vector_max_q_per_a - vector_prev_q_per_a), axis=[0, 1])

            return - 2 * q_grad

        self._b_network = self._network["model"]["net"][0]
        self._a_network = self._network["model"]["net"][1]
        self._c_network = self._network["model"]["net"][2]

        self._b_parameters = self._network["model"]["params"][0]
        self._a_parameters = self._network["model"]["params"][1]
        self._c_parameters = self._network["model"]["params"][2]

        self._q_planning_loss_grad = jax.jit(q_planning_grad)

        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [0, 1, 2], has_aux=True))
        # self._model_loss_grad = jax.value_and_grad(model_loss, [0, 1, 2], has_aux=True)
        self._a_forward = jax.jit(self._a_network)
        self._b_forward = jax.jit(self._b_network)
        self._c_forward = jax.jit(self._c_network)
        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model,
                                                                self._exploration_decay_period, 0, 0.9)
        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
        self._model_opt_update = jax.jit(model_opt_update)
        self._model_opt_state = model_opt_init([self._a_parameters, self._b_parameters, self._c_parameters])
        self._model_get_params = model_get_params


    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            (total_loss, losses), gradients = self._model_loss_grad(self._a_parameters,
                                                   self._b_parameters,
                                                   self._c_parameters,
                                                  self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._a_parameters, self._b_parameters, self._c_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_cross(A)": losses["cross_loss(A)"],
                "loss_o": losses["expected_loss(b)"],
                "loss_r": losses["r_loss(w_r)"],
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False


    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if self._n == 0:
            return
        if timestep.discount is None:
            return
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
        d_t = np.array([timestep.discount])

        gradient = self._q_planning_loss_grad(self._q_parameters,
                                                    self._a_parameters,
                                                    self._b_parameters,
                                                    self._c_parameters,
                                                    o_t, d_t)
        self._q_opt_state = self._q_opt_update(self.episode, gradient,
                                               self._q_opt_state)
        self._q_parameters = self._q_get_params(self._q_opt_state)

        losses_and_grads = {"losses": {},
                            "gradients": {"grad_norm_q_planning": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        self._log_summaries(losses_and_grads, "q_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        features = self._get_features([timestep.observation])
        next_features = self._get_features([new_timestep.observation])
        transitions = [np.array(features),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array(next_features)]

        self._sequence.append(transitions)

        if new_timestep.discount == 0:
            self._should_reset_sequence = True

