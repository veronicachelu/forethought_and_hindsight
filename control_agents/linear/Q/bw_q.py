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


class BwQ(VanillaQ):
    def __init__(
            self,
            **kwargs
    ):
        super(BwQ, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        def model_loss(o_params,
                       r_params,
                       transitions):
            o_tmn = transitions[0][0]
            a_tmn = transitions[0][1]
            o_t = transitions[-1][-1]
            model_o_tmn = jax.vmap(lambda
                                     model_otmn, a:
                                   model_otmn[a])(self._o_network(o_params, o_t), a_tmn)

            o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_tmn, o_tmn))

            # r_input = jnp.concatenate([model_o_tmn, o_t], axis=-1)
            # model_r_tmn = self._r_network(r_params, lax.stop_gradient(r_input))
            model_r_tmn = self._r_network(r_params, o_t)
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn, r_t_target))
            total_loss = o_loss + r_loss

            return total_loss, {"o_loss": o_loss,
                               "r_loss": r_loss,
                               }

        def q_planning_loss(q_params, o_params, r_params, x, d):
            # tile the input observation x nA
            x_per_a = jnp.tile(jnp.expand_dims(x, 1), (1, self._nA, 1))
            # tile gamma x nA
            d_per_a = jnp.tile(jnp.expand_dims(d, 1), (1, self._nA))

            # model output nA x nF
            prev_x_per_a = self._o_forward(o_params, x)
            b, na, nf = prev_x_per_a.shape
            # flatten predecessor nA * nF
            prev_x = jnp.reshape(prev_x_per_a, (-1, nf))
            # flatten current nA * nF
            x_flat = jnp.reshape(x_per_a, (-1, nf))
            # r_input_flat = jnp.concatenate([prev_x, x_flat], axis=-1)
            # flattened reward nA * nF
            # r_t_flat = self._r_forward(r_params, r_input_flat)
            r_t_flat = self._r_forward(r_params, x_flat)
            # reward nA x nF
            r_per_a = jnp.reshape(r_t_flat, (b, na))

            # q at all flatten predecessor states and actions x
            prev_q = self._q_forward(q_params, prev_x)
            # all actions in batch
            a_per_a = jnp.tile(jnp.expand_dims(jnp.arange(self._nA), 0),
                               (b, 1))
            the_as = jnp.reshape(a_per_a, (-1))
            prev_q = jax.vmap(lambda q, a: q[a], in_axes=(0, 0), out_axes=(0))(prev_q, the_as)
            prev_q_per_a = jnp.reshape(prev_q, (b, na))

            q = jnp.max(self._q_forward(q_params, x), axis=-1)
            q_per_a = jnp.tile(jnp.expand_dims(q, 1), (1, self._nA))

            target_tm1 = r_per_a + d_per_a * jnp.array([self._discount ** self._n]) * q_per_a
            td_errors_per_a = jax.lax.stop_gradient(target_tm1) - prev_q_per_a

            loss = jnp.mean(td_errors_per_a ** 2)

            return loss

        self._o_network = self._network["model"]["net"][0]
        self._r_network = self._network["model"]["net"][2]

        self._o_parameters = self._network["model"]["params"][0]
        self._r_parameters = self._network["model"]["params"][2]

        # self._q_planning_loss_grad = jax.jit(jax.value_and_grad(q_planning_loss, 0))
        self._q_planning_loss_grad = jax.value_and_grad(q_planning_loss, 0)

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

    def get_model_for_all_states(self, all_states):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        return np.array(self._o_forward(self._o_parameters, np.asarray(features)), np.float)

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
            prev_timestep=None
    ):
        if self._n == 0:
            return
        if timestep.discount is None:
            return
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
        d_t = np.array([timestep.discount])

        loss, gradient = self._q_planning_loss_grad(self._q_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    o_t, d_t)
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



