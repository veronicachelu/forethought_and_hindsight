import os
from typing import Any
from typing import Callable, Sequence

import dm_env
import jax
import numpy as np
import tensorflow as tf
from jax import lax
from jax import numpy as jnp
from jax.experimental import optimizers

from agents.linear.MLE.lp_vanilla import LpVanilla
import rlax
from basis.feature_mapper import FeatureMapper
from utils.visualizer import *

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class LpBwFw(LpVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpBwFw, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        self._bw_gain = np.zeros((100))
        self._td_gain = np.zeros((100))
        self._fw_gain = np.zeros((100))

        def model_loss(bw_o_params,
                       fw_o_params,
                       bw_r_params,
                       fw_r_params,
                       transitions):
            o_tmn = transitions[0][0]
            o_t = transitions[-1][-1]
            model_o_tmn = self._bw_o_network(bw_o_params, o_t)
            model_o_t = self._fw_o_network(fw_o_params, o_tmn)

            bw_o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_tmn, o_tmn))
            fw_o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_t, o_t))

            bw_r_input = jnp.concatenate([model_o_tmn, o_t], axis=-1)
            fw_r_input = jnp.concatenate([o_tmn, model_o_t], axis=-1)
            model_bw_r = self._bw_r_network(bw_r_params, lax.stop_gradient(bw_r_input))
            model_fw_r = self._fw_r_network(fw_r_params, lax.stop_gradient(fw_r_input))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            bw_r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_bw_r, r_t_target))
            fw_r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_fw_r, r_t_target))
            total_loss = bw_o_loss + fw_o_loss + bw_r_loss + fw_r_loss
                         #+ self._alpha_reg1 * l1_reg + \
                         #self._alpha_reg2 * l2_reg

            return total_loss, {
                "bw_o_loss": bw_o_loss,
                "fw_o_loss": fw_o_loss,
                "bw_r_loss": bw_r_loss,
                "fw_r_loss": fw_r_loss,
                                # "reg1": l1_reg,
                                # "reg2": l2_reg
                               }

        def bw_v_planning_loss(v_params, bw_o_params, bw_r_params, o_t, d_t):
            o_tmn = self._bw_o_forward(bw_o_params, o_t)
            v_tmn = jnp.squeeze(self._v_network(v_params, lax.stop_gradient(o_tmn)), axis=-1)
            r_input = jnp.concatenate([o_tmn, o_t], axis=-1)

            r_tmn = self._bw_r_forward(bw_r_params, r_input)
            v_t_target = jnp.squeeze(self._v_network(v_params, o_t), axis=-1)

            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn,
                                                  d_t * jnp.array([self._discount ** self._n]),
                                                 v_t_target)
            return jnp.mean(td_error ** 2)

        def fw_v_planning_loss(v_params, fw_o_params, fw_r_params, o_tmn):
            o_t = self._fw_o_forward(fw_o_params, o_tmn)
            v_tmn = jnp.squeeze(self._v_network(v_params, o_tmn), axis=-1)
            r_input = jnp.concatenate([o_tmn, o_t], axis=-1)

            r_tmn = self._fw_r_forward(fw_r_params, r_input)
            v_t_target = jnp.squeeze(self._v_network(v_params, o_t), axis=-1)

            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn,
                                                  jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        self._bw_o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        self._bw_r_network = self._network["model"]["net"][2]
        self._fw_r_network = self._network["model"]["net"][3]

        self._bw_o_parameters = self._network["model"]["params"][0]
        self._fw_o_parameters = self._network["model"]["params"][1]
        self._bw_r_parameters = self._network["model"]["params"][2]
        self._fw_r_parameters = self._network["model"]["params"][3]

        self._bw_v_planning_loss_grad = jax.jit(jax.value_and_grad(bw_v_planning_loss, 0))
        self._fw_v_planning_loss_grad = jax.jit(jax.value_and_grad(fw_v_planning_loss, 0))

        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [0, 1, 2, 3], has_aux=True))
        self._bw_o_forward = jax.jit(self._bw_o_network)
        self._fw_o_forward = jax.jit(self._fw_o_network)
        self._bw_r_forward = jax.jit(self._bw_r_network)
        self._fw_r_forward = jax.jit(self._fw_r_network)
        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model,
                                                                self._exploration_decay_period, 0, 0.9)
        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
        self._model_opt_update = jax.jit(model_opt_update)
        self._model_opt_state = model_opt_init([self._bw_o_parameters,
                                                self._fw_o_parameters,
                                                self._bw_r_parameters,
                                                self._fw_r_parameters,
                                                ])
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
            (total_loss, losses), gradients = self._model_loss_grad(self._bw_o_parameters,
                                                                    self._fw_o_parameters,
                                                                    self._bw_r_parameters,
                                                                    self._fw_r_parameters,
                                                                    self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._bw_o_parameters,\
            self._fw_o_parameters,\
            self._bw_r_parameters,\
            self._fw_r_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_fw_o": losses["fw_o_loss"],
                "loss_bw_o": losses["bw_o_loss"],
                "loss_bw_r": losses["bw_r_loss"],
                "loss_fw_r": losses["fw_r_loss"],
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
            rmsve,
            environment,
            mdp_solver,
            space
    ):
        if self._n == 0:
            return

        bw_v_parameters = self.bw_planning_update(timestep)
        fw_v_parameters = self.fw_planning_update(timestep)
        if bw_v_parameters is not None and fw_v_parameters is not None:
            true_v = mdp_solver.get_optimal_v()
            hat_bw_v = self.get_values_for_all_states(environment.get_all_states(), bw_v_parameters)
            hat_fw_v = self.get_values_for_all_states(environment.get_all_states(), fw_v_parameters)
            hat_v = self.get_values_for_all_states(environment.get_all_states(), self._v_parameters)
            _hat_v_ = environment.reshape_v(hat_v * (environment._d * len(environment._starting_positions)))
            _hat_bw_v_ = environment.reshape_v(hat_bw_v * (environment._d * len(environment._starting_positions)))
            _hat_fw_v_ = environment.reshape_v(hat_fw_v * (environment._d * len(environment._starting_positions)))
            _true_v = environment.reshape_v(
                true_v * environment._d * len(environment._starting_positions))
            # plot_v(env=environment,
            #        values=_hat_v_,
            #        logs=self._images_dir,
            #        true_v=_true_v,
            #        env_type=space["env_config"]["env_type"],
            #        filename="v_{}_{}.png".format(self.episode, self.total_steps))
            # plot_v(env=environment,
            #        values=_hat_bw_v_,
            #        logs=self._images_dir,
            #        true_v=_true_v,
            #        env_type=space["env_config"]["env_type"],
            #        filename="bw_v_{}_{}.png".format(self.episode, self.total_steps))
            # plot_v(env=environment,
            #        values=_hat_fw_v_,
            #        logs=self._images_dir,
            #        true_v=_true_v,
            #        env_type=space["env_config"]["env_type"],
            #        filename="fw_v_{}_{}.png".format(self.episode, self.total_steps))
            # plot_v(env=environment,
            #        values=_true_v,
            #        logs=self._images_dir,
            #        true_v=_true_v,
            #        env_type=space["env_config"]["env_type"],
            #        filename="true_v_{}_{}.png".format(self.episode, self.total_steps))

            # rmsve = np.sqrt(np.sum(environment._d * ((true_v - hat_v) ** 2)))
            rmsve_bw = np.sqrt(np.sum(environment._d * ((true_v - hat_bw_v) ** 2)))
            rmsve_fw = np.sqrt(np.sum(environment._d * ((true_v - hat_fw_v) ** 2)))

            bw_gain = rmsve - rmsve_bw
            fw_gain = rmsve - rmsve_fw

            self._bw_gain[np.ravel_multi_index(timestep.observation, (10, 10))] += bw_gain
            self._fw_gain[np.ravel_multi_index(timestep.observation, (10, 10))] += fw_gain


    def bw_planning_update(self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):

        if timestep.discount is None:
            return None
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
        d_t = np.array(timestep.discount)

        # plan on batch of transitions
        loss, gradient = self._bw_v_planning_loss_grad(self._v_parameters,
                                                    self._bw_o_parameters,
                                                    self._bw_r_parameters,
                                                    o_t, d_t)
        bw_v_opt_state = self._v_opt_update(self.episode, gradient,
                                               self._v_opt_state)
        bw_v_parameters = self._v_get_params(bw_v_opt_state)

        return bw_v_parameters

    def fw_planning_update(self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if timestep.last():
            return None
        features = self._get_features([timestep.observation])
        o_t = np.array(features)

        # plan on batch of transitions
        loss, gradient = self._fw_v_planning_loss_grad(self._v_parameters,
                                                    self._fw_o_parameters,
                                                    self._fw_r_parameters,
                                                    o_t)
        fw_v_opt_state = self._v_opt_update(self.episode, gradient,
                                               self._v_opt_state)
        fw_v_parameters = self._v_get_params(fw_v_opt_state)

        return fw_v_parameters

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def load_model(self):
        return
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._v_parameters = to_load["v_parameters"]
            self._o_parameters = to_load["o_parameters"]
            self._r_parameters = to_load["r_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        return
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "v_parameters": self._v_parameters,
            "o_parameters": self._o_parameters,
            "r_parameters": self._r_parameters,
        }
        np.save(checkpoint, to_save)
        print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))

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

    def _log_summaries(self, losses_and_grads, summary_name):
        if self._logs is not None:
            losses = losses_and_grads["losses"]
            if "gradients" in losses_and_grads.keys():
                gradients = losses_and_grads["gradients"]
            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k),
                                      np.array(v), step=ep)
                if "gradients" in losses_and_grads.keys():
                    for k, v in gradients.items():
                        tf.summary.scalar("train/gradients/{}/{}".format(summary_name, k),
                                          gradients[k], step=ep)
                self.writer.flush()

    def get_values_for_all_states(self, all_states, params):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        return np.array(np.squeeze(self._v_forward(params, np.array(features)), axis=-1), np.float)
