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

from agents.linear.extrinsic.lp_vanilla import LpVanilla
import rlax

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class LpFw(LpVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpFw, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        def model_loss(fw_o_online_params,
                       r_online_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]
            model_o_t = self._fw_o_network(fw_o_online_params, o_tmn_target)

            fw_o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_t, o_t))

            r_input = jnp.concatenate([o_tmn_target, model_o_t], axis=-1)
            model_r_tmn = self._r_network(r_online_params, lax.stop_gradient(r_input))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn, r_t_target))
            total_loss = fw_o_loss + r_loss

            return total_loss, {"fw_o_loss": fw_o_loss,
                                "r_loss": r_loss
                                }

        def v_planning_loss(v_params, fw_o_params, r_params, o_tmn, d_t):
            o_t = self._fw_o_forward(fw_o_params, o_tmn)
            v_tmn = self._v_network(v_params, o_tmn)
            r_input = jnp.concatenate([o_tmn, o_t], axis=-1)

            r_tmn = self._r_forward(r_params, r_input)
            v_t_target = self._v_network(v_params, o_t)
            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn,
                                                  d_t * jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, 0))
        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model,
                                                                self._exploration_decay_period, 0, 0.9)
        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [0, 1], has_aux=True))
        self._fw_o_forward = jax.jit(self._fw_o_network)
        self._r_forward = jax.jit(self._r_network)

        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
        self._model_opt_update = jax.jit(model_opt_update)
        self._model_opt_state = model_opt_init([self._fw_o_parameters, self._r_parameters])
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
            (total_loss, losses), gradients = self._model_loss_grad(self._fw_o_parameters,
                                                                    self._r_parameters,
                                                                    self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._fw_o_parameters, self._r_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_fw_o": losses["fw_o_loss"],
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
        o_t = np.array([features])
        d_t = np.array([timestep.discount])

        # plan on batch of transitions
        loss, gradient = self._v_planning_loss_grad(self._v_parameters,
                                                    self._fw_o_parameters,
                                                    self._r_parameters,
                                                    o_t, d_t)
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

    def load_model(self):
        return
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._v_parameters = to_load["v_parameters"]
            self._fw_o_parameters = to_load["o_parameters"]
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
            "o_parameters": self._fw_o_parameters,
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
