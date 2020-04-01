from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs
from jax import grad

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import numpy as np
from prediction_agents.linear.vanillar_linear_prediction import VanillaLinearPrediction
import tensorflow as tf

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class nStepLpJumpyExp(VanillaLinearPrediction):
    def __init__(
            self,
            **kwargs
    ):
        super(nStepLpJumpyExp, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        def model_loss(o_online_params,
                       r_online_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]
            model_o_tmn = self._o_network(o_online_params, o_t)

            o_error = model_o_tmn - \
                      lax.stop_gradient(o_tmn_target)
                      # (self._discount ** self._n) * lax.stop_gradient(o_tmn_target)
            o_error = jnp.mean(o_error ** 2)

            model_r_tmn = self._r_network(r_online_params, model_o_tmn)
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_error = model_r_tmn - lax.stop_gradient(r_t_target)
            r_error = jnp.mean(r_error ** 2)

            total_error = o_error + r_error
            return total_error

        def v_planning_loss(v_params, o_params, r_params, o_t):
            o_tmn = lax.stop_gradient(self._o_forward(o_params, o_t))

            v_tmn = self._v_network(v_params, o_tmn)
            r_tmn = self._r_forward(r_params, o_tmn)

            td_error = v_tmn - lax.stop_gradient(r_tmn + (self._discount ** self._n) *
                                                 self._v_network(v_params, o_t))

            return jnp.mean(td_error ** 2)

        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, 0))

        # This function computes dL/dTheta
        self._o_loss_grad = jax.jit(jax.value_and_grad(model_loss, 0))
        self._r_loss_grad = jax.jit(jax.value_and_grad(model_loss, 1))
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)

        # Make an Adam optimizer.
        # self._o_step_schedule = optimizers.inverse_time_decay(self._lr_model,
        #                                                       self._exploration_decay_period
        #                                                       * 17,
        #                                                       100.0)
        self._o_step_schedule = self._lr_model
        o_opt_init, o_opt_update, o_get_params = optimizers.adam(step_size=self._o_step_schedule)
        # o_opt_init, o_opt_update, o_get_params = optimizers.adam(step_size=self._lr_model)
        self._o_opt_update = jax.jit(o_opt_update)
        self._o_opt_state = o_opt_init(self._o_parameters)
        self._o_get_params = o_get_params
        # self._r_step_schedule = optimizers.inverse_time_decay(self._lr_model,
        #                                                       self._exploration_decay_period
        #                                                       * 17,
        #                                                       100.0)
        self._r_step_schedule = self._lr_model
        r_opt_init, r_opt_update, r_get_params = optimizers.adam(step_size=self._r_step_schedule)
        # r_opt_init, r_opt_update, r_get_params = optimizers.adam(step_size=self._lr_model)
        self._r_opt_update = jax.jit(r_opt_update)
        self._r_opt_state = r_opt_init(self._r_parameters)
        self._r_get_params = r_get_params

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        return np.argmax(self._pi[np.argmax(timestep.observation)])

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def load_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._v_parameters = to_load["v_parameters"]
            self._o_parameters = to_load["o_parameters"]
            self._r_parameters = to_load["r_parameters"]
            self._d_parameters = to_load["d_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "v_parameters": self._v_parameters,
            "o_parameters": self._o_parameters,
            "r_parameters": self._r_parameters,
            "d_parameters": self._d_parameters,
        }
        np.save(checkpoint, to_save)
        print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            loss_o, gradient_o = self._o_loss_grad(self._o_parameters,
                                                  self._r_parameters,
                                                  self._sequence)
            self._o_opt_state = self._o_opt_update(self.total_steps, gradient_o,
                                                           self._o_opt_state)
            self._o_parameters = self._o_get_params(self._o_opt_state)

            loss_r, gradient_r = self._r_loss_grad(self._o_parameters,
                                               self._r_parameters,
                                               self._sequence)
            self._r_opt_state = self._r_opt_update(self.total_steps, gradient_r,
                                                   self._r_opt_state)
            self._r_parameters = self._r_get_params(self._r_opt_state)

            losses_and_grads = {"losses": {
                "loss_o": loss_o,
                "loss_r": loss_r,
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
    ):
        if self._n == 0:
            return
        o_t = np.array([timestep.observation])

        # plan on batch of transitions
        loss, gradient = self._v_planning_loss_grad(self._v_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    o_t)
        self._v_opt_state = self._v_opt_update(self.total_steps, gradient,
                                               self._v_opt_state)
        self._v_parameters = self._v_get_params(self._v_opt_state)

        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
                                       },
                            "gradients": {"grad_norm_v_planning": np.sum(
                                np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        self._log_summaries(losses_and_grads, "value_planning")

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        self._sequence.append([np.array([timestep.observation]),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array([new_timestep.observation])])
        if new_timestep.discount == 0:
            self._should_reset_sequence = True

    def _log_summaries(self, losses_and_grads, summary_name):
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

    def get_values_for_all_states(self, all_states):
        return np.array(self._v_forward(self._v_parameters, np.array(all_states)), np.float)
