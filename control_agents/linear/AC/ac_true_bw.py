from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs
from rlax._src import distributions
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
from control_agents.linear.AC.ac_vanilla import ACVanilla

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class ACTrueBw(ACVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(ACTrueBw, self).__init__(**kwargs)

        self._sequence_model = []
        self._should_reset_sequence = False

        def model_loss(a_params,
                       b_params,
                       c_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            model_expected_o_tmn = self._b_network(b_params, o_t)
            model_cross_o_tmn = self._a_network(a_params, o_t)

            cross_o_tmn_target = lax.batch_matmul(o_tmn_target[..., None],
                                                  jnp.transpose(o_tmn_target[..., None],
                                                                axes=[0, 2, 1]))
            # cross_o_tmn_target - jax.numpy.diagflat(model_expected_o_tmn)
            # wrong_cross_o_tmn = lax.batch_matmul(model_expected_o_tmn[..., None],
            #                                jnp.transpose(model_expected_o_tmn[..., None], axes=[0, 2, 1]))
            b_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_expected_o_tmn, o_tmn_target))
            a_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_cross_o_tmn, cross_o_tmn_target))
            # distance_from_wrong_cross = jnp.sum(jax.vmap(rlax.l2_loss)(model_cross_o_tmn, wrong_cross_o_tmn))

            model_vector_r_tmn = self._c_network(c_params,
                                                 (lax.stop_gradient(model_cross_o_tmn),
                                                  lax.stop_gradient(model_expected_o_tmn),
                                                  o_t))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]
            vector_r_t_target = o_tmn_target * r_t_target[..., None]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_vector_r_tmn, vector_r_t_target))
            total_loss = b_loss + r_loss + a_loss

            return total_loss, {"cross_loss(A)": a_loss,
                                "expected_loss(b)": b_loss,
                                "r_loss(w_r)": r_loss,
                                }

        def v_planning_grad(v_params, a_params, b_params, c_params, o_t,
                            d_t):
            cross_o_tmn = self._a_forward(a_params, o_t)
            expected_o_tmn = self._b_forward(b_params, o_t)

            vector_r_tmn = self._c_network(c_params,
                                           (cross_o_tmn, expected_o_tmn, o_t))[..., None]

            v_error = d_t * jnp.array([self._discount ** self._n]) * \
                      lax.batch_matmul(expected_o_tmn[..., None],
                                       jnp.transpose(o_t[..., None], axes=[0, 2, 1]))\
                      - cross_o_tmn

            v_model_error = lax.batch_matmul(v_error, v_params[None, ...])
            v_grad = jnp.mean(0.5 * (vector_r_tmn + v_model_error), axis=0)

            return -0.5 * v_grad

        # Internalize the networks.
        self._v_network = self._network["value"]["net"]
        self._v_parameters = self._network["value"]["params"]

        self._pi_network = self._network["pi"]["net"]
        self._pi_parameters = self._network["pi"]["params"]

        self._h_network = self._network["model"]["net"][0]
        self._b_network = self._network["model"]["net"][1]
        self._a_network = self._network["model"]["net"][2]
        self._c_network = self._network["model"]["net"][3]

        self._h_parameters = self._network["model"]["params"][0]
        self._b_parameters = self._network["model"]["params"][1]
        self._a_parameters = self._network["model"]["params"][2]
        self._c_parameters = self._network["model"]["params"][3]

        self._v_planning_loss_grad = jax.jit(v_planning_grad)
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

    def _get_features(self, o):
        if self._feature_mapper is not None:
            return self._feature_mapper.get_features(o, self._nrng)
        else:
            return o

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        features = self._get_features(timestep.observation[None, ...])
        pi_logits = self._pi_forward(self._pi_parameters, features)
        if eval:
            action = np.argmax(pi_logits, axis=-1)[0]
        else:
            key = next(self._rng_seq)
            action = jax.random.categorical(key, pi_logits).squeeze()
            # print(np.argmax(pi_logits, axis=-1))
        return int(action)

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence_model) >= self._n:
            (total_loss, losses), gradients = self._model_loss_grad(self._a_parameters,
                                                                    self._b_parameters,
                                                                    self._c_parameters,
                                                                    self._sequence_model)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._a_parameters, self._b_parameters, self._c_parameters = self._model_parameters

            self._o_parameters_norm = np.linalg.norm(self._a_parameters[0], 2) + \
                                      np.linalg.norm(self._b_parameters, 2)
            self._r_parameters_norm = np.linalg.norm(self._c_parameters[0], 2)

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_cross(A)": losses["cross_loss(A)"],
                "loss_o": losses["expected_loss(b)"],
                "L2_norm_o": self._o_parameters_norm,
                "L2_norm_r": self._r_parameters_norm,
                "loss_r": losses["r_loss(w_r)"],
                # "loss_distance_A": losses["distance_from_wrong_cross"],
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence_model = self._sequence_model[1:]

        if self._should_reset_sequence:
            self._sequence_model = []
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
        d_t = np.array(timestep.discount)

        gradient = self._v_planning_loss_grad(self._v_parameters,
                                              self._a_parameters,
                                              self._b_parameters,
                                              self._c_parameters,
                                              o_t, d_t)
        self._v_opt_state = self._v_opt_update(self.episode, gradient,
                                               self._v_opt_state)
        self._v_parameters = self._v_get_params(self._v_opt_state)

        losses_and_grads = {"losses": {},
                            "gradients": {"grad_norm_v_planning": np.sum(
                                np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        self._log_summaries(losses_and_grads, "value_planning")


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

        self._sequence.append([features[0],
                       action,
                       new_timestep.reward,
                       new_timestep.discount,
                       next_features[0]])

        self._sequence_model.append([np.array(features),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array(next_features)])

        if new_timestep.discount == 0:
            self._should_reset_sequence = True

    def _log_summaries(self, losses_and_grads, summary_name):
        if self._logs is not None:
            losses = losses_and_grads["losses"]
            # gradients = losses_and_grads["gradients"]
            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), np.array(v), step=ep)
                # for k, v in gradients.items():
                #     tf.summary.scalar("train/gradients/{}/{}".format(summary_name, k),
                #                       gradients[k], step=ep)
                self.writer.flush()

    # def get_values_for_all_states(self, all_states):
    #     features = self._get_features(all_states) if self._feature_mapper is not None else all_states
    #     latents = self._h_forward(self._h_parameters, np.array(features)) if self._latent else features
    #     return np.array(self._v_forward(self._v_parameters, np.asarray(latents, np.float)), np.float)

    def get_policy_for_all_states(self, all_states):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        pi_logits = self._pi_forward(self._pi_parameters, features)
        actions = np.argmax(pi_logits, axis=-1)

        return np.array(actions)

    def get_values_for_all_states(self, all_states):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        return np.array(np.squeeze(self._v_forward(self._v_parameters, np.array(features)), axis=-1), np.float)
    # def update_hyper_params(self, episode, total_episodes):
    #     steps_left = self._exploration_decay_period - episode
    #     bonus = (1.0 - self._epsilon) * steps_left / self._exploration_decay_period
    #     bonus = np.clip(bonus, 0., 1. - self._epsilon)
    #     self._epsilon = self._epsilon + bonus

    def update_hyper_params(self, episode, total_episodes):
        # decay_period, step, warmup_steps, epsilon):
        steps_left = total_episodes + 0 - episode
        bonus = (self._initial_epsilon - self._final_epsilon) * steps_left / total_episodes
        bonus = np.clip(bonus, 0., self._initial_epsilon - self._final_epsilon)
        self._epsilon = self._final_epsilon + bonus
        if self._logs is not None:
            # if self._max_len == -1:
            ep = self.total_steps
            # else:
            #     ep = self.episode
            if ep % self._log_period == 0:
                tf.summary.scalar("train/epsilon",
                                  self._epsilon, step=ep)
                self.writer.flush()
