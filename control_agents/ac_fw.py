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
from control_agents.ac_vanilla import ACVanilla

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class ACFw(ACVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(ACFw, self).__init__(**kwargs)

        self._sequence_model = []
        self._should_reset_sequence = False

        def model_loss(o_params,
                       r_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            model_o_t = self._o_network(o_params, o_tmn_target)

            fw_o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_t, o_t))

            r_input = jnp.concatenate([o_tmn_target, model_o_t], axis=-1)
            model_r_tmn = self._r_network(r_params, lax.stop_gradient(r_input))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn, r_t_target))
            total_loss = fw_o_loss + r_loss

            return total_loss, {"fw_o_loss": fw_o_loss,
                                "r_loss": r_loss,
                                }

        def v_planning_loss(v_params, o_params, r_params, o_tmn):
            o_t = self._o_forward(o_params, o_tmn)
            v_tmn = jnp.squeeze(self._v_network(v_params, o_tmn), axis=-1)
            r_input = jnp.concatenate([o_tmn, o_t], axis=-1)

            r_tmn = self._r_forward(r_params, r_input)
            v_t_target = jnp.squeeze(self._v_network(v_params, o_t), axis=-1)

            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn,
                                                  jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        # Internalize the networks.
        self._v_network = self._network["value"]["net"]
        self._v_parameters = self._network["value"]["params"]

        self._pi_network = self._network["pi"]["net"]
        self._pi_parameters = self._network["pi"]["params"]

        self._h_network = self._network["model"]["net"][0]
        self._o_network = self._network["model"]["net"][1]
        # self._fw_o_network = network["model"]["net"][2]
        self._r_network = self._network["model"]["net"][3]

        self._h_parameters = self._network["model"]["params"][0]
        self._o_parameters = self._network["model"]["params"][1]
        # self._fw_o_parameters = network["model"]["params"][2]
        self._r_parameters = self._network["model"]["params"][3]

        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, 0))

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
        if timestep.last():
            return
        if self._n == 0:
            return
        if len(self._sequence_model) >= self._n:
            (total_loss, losses), gradients = self._model_loss_grad(self._o_parameters,
                                                                    self._r_parameters,
                                                                    self._sequence_model)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._o_parameters, self._r_parameters = self._model_parameters

            self._o_parameters_norm = np.linalg.norm(self._o_parameters, 2)
            self._r_parameters_norm = np.linalg.norm(self._r_parameters[0], 2)

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_o": losses["fw_o_loss"],
                "L2_norm_o": self._o_parameters_norm,
                "L2_norm_r": self._r_parameters_norm,
                "loss_r": losses["r_loss"],
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
        if timestep.last():
            return
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
        # d_t = np.array([timestep.discount])

        # plan on batch of transitions
        loss, gradient = self._v_planning_loss_grad(self._v_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    o_t)
        self._v_opt_state = self._v_opt_update(self.episode, gradient,
                                               self._v_opt_state)
        self._v_parameters = self._v_get_params(self._v_opt_state)

        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
                                       },
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
        features = self._get_features([timestep.observation])[0]
        next_features = self._get_features([new_timestep.observation])[0]
        transitions = [features,
                       action,
                       new_timestep.reward,
                       new_timestep.discount,
                       next_features]

        self._sequence.append(transitions)

        transitions = [np.array(features),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array(next_features)]

        self._sequence_model.append(transitions)

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
