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

from control_agents.linear.Q.vanilla_q import VanillaQ
NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]
import functools

class FwQ(VanillaQ):
    def __init__(
            self,
            **kwargs
    ):
        super(FwQ, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        def model_loss(o_params,
                       r_params,
                       transitions):
            o_tmn = transitions[0][0]
            a_tmn = transitions[0][1]
            o_t = transitions[-1][-1]
            model_o_t = jax.vmap(lambda
                                 model_ot, a:
                                 model_ot[a])(self._o_network(o_params, o_tmn), a_tmn)

            o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_t, o_t))

            r_input = jnp.concatenate([o_tmn, model_o_t], axis=-1)
            model_r_tmn = self._r_network(r_params, lax.stop_gradient(r_input))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn, r_t_target))
            total_loss = o_loss + r_loss

            return total_loss, {"o_loss": o_loss,
                               "r_loss": r_loss,
                               }

        def q_planning_loss(q_params, o_params, r_params, x):
            x_per_a = jnp.tile(jnp.expand_dims(x, 1), (1, self._nA, 1))
            next_x_per_a = self._o_network(o_params, x)
            b, na, nf = next_x_per_a.shape
            next_x = jnp.reshape(next_x_per_a, (-1, nf))
            x_flat = jnp.reshape(x_per_a, (-1, nf))
            r_input_flat = jnp.concatenate([x_flat, next_x], axis=-1)
            r_t_flat = self._r_forward(r_params, r_input_flat)
            r_per_a = jnp.reshape(r_t_flat, (b, na))

            # r_per_a = jax.vmap(functools.partial(self._r_forward_per_action,
            #                                      r_params=r_params),
            #                    in_axes=(1, 1), out_axes=(1))(next_x_per_a, x_per_a)

            next_q = self._q_forward(q_params, next_x)
            next_q = jnp.reshape(next_q, (b, na, na))
            max_next_q_per_a = jnp.max(next_q, axis=-1)
            # next_q_per_a = jax.vmap(functools.partial(self._q_forward_per_action,
            #                                      q_params=q_params),
                               # in_axes=(1),
                               # out_axes=(1))(next_x_per_a)
            q_per_a = self._q_forward(q_params, x)

            target_tm1 = r_per_a + jnp.array([self._discount ** self._n]) * max_next_q_per_a
            td_errors_per_a = jax.lax.stop_gradient(target_tm1) - q_per_a

            # td_errors_per_a = jax.vmap(self._q_planning_per_action,
            #          in_axes=(1, 1, 1),
            #          out_axes=(1))(q_per_a, r_per_a, next_q_per_a)
            # td_errors = jax.vmap(lambda td_error, a: td_error[a], in_axes=(0, 0))(td_errors_per_a, a)
            # td_errors = jax.vmap(lambda td_error, a: td_error[a], in_axes=(0, 0))(td_errors_per_a, a)

            loss = jnp.mean(td_errors_per_a ** 2)

            return loss

        self._o_network = self._network["model"]["net"][0]
        self._r_network = self._network["model"]["net"][2]

        self._o_parameters = self._network["model"]["params"][0]
        self._r_parameters = self._network["model"]["params"][2]

        self._q_planning_loss_grad = jax.jit(jax.value_and_grad(q_planning_loss, 0))
        # self._q_planning_loss_grad = jax.value_and_grad(q_planning_loss, 0)

        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [0, 1], has_aux=True))
        # self._model_loss_grad = jax.value_and_grad(model_loss, [0, 1], has_aux=True)
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)
        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model,
                                                                self._exploration_decay_period, 0, 0.9)
        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
        self._model_opt_update = jax.jit(model_opt_update)
        self._model_opt_state = model_opt_init([self._o_parameters, self._r_parameters])
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
            (total_loss, losses), gradients = self._model_loss_grad(self._o_parameters,
                                                   self._r_parameters,
                                                  self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._o_parameters, self._r_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_o": losses["o_loss"],
                "loss_r": losses["r_loss"],
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
            # action,
            prev_timestep=None
    ):
        if timestep.last():
            return
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
        # a_t = np.array(action)

        loss, gradient = self._q_planning_loss_grad(self._q_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    o_t)
        self._q_opt_state = self._q_opt_update(self.episode, gradient,
                                               self._q_opt_state)
        self._q_parameters = self._q_get_params(self._q_opt_state)

        losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
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

