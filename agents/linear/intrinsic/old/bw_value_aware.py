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
import rlax
from jax.experimental import stax
import itertools

from agents.linear.intrinsic.lp_intrinsic_vanilla import LpIntrinsicVanilla

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]

def td_learning(
    v_tm1,
    r_t,
    discount_t,
    v_t):

  target_tm1 = r_t + discount_t * lax.stop_gradient(v_t)
  return target_tm1 - v_tm1

class LpBWValueAware(LpIntrinsicVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpBWValueAware, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._target_update_period = 1

        def latent_v_loss(v_params, h_params, transitions):
            h_tm1, _, r_t, d_t, h_t = transitions
            v_tm1 = self._v_network(v_params, h_tm1)
            v_t = self._v_network(v_params, h_t)
            td_error = jax.vmap(td_learning)(v_tm1, r_t, d_t * (self._discount ** self._n), v_t)
            return jnp.mean(td_error ** 2)

        def model_loss(v_params,
                       h_params,
                       o_params,
                       r_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            h_tmn = self._h_network(h_params, o_tmn_target) if self._latent else o_tmn_target
            h_t = self._h_network(h_params, o_t) if self._latent else o_t

            real_r_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_r_tmn_2_t += (self._discount ** i) * t[2]

            real_d_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_d_tmn_2_t += t[3]

            real_transitions = [o_tmn_target,
                           jnp.array([0]),
                           real_r_tmn_2_t,
                           real_d_tmn_2_t,
                                h_t]

            _, real_gradients = self._latent_v_loss_grad(v_params,
                                                self._h_parameters,
                                                  real_transitions)
            real_v_opt_state = self._v_opt_update(self.episode, real_gradients,
                                                  lax.stop_gradient(self._v_opt_state))
            real_value_params = self._v_get_params(real_v_opt_state)

            model_tmn = self._o_network(o_params, h_t)
            model_r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            model_r_tmn_2_t = self._r_network(r_params, model_r_input)

            model_transitions = [model_tmn,
                           jnp.array([0]),
                           model_r_tmn_2_t,
                           real_d_tmn_2_t,
                           h_t]

            _, model_gradients = self._latent_v_loss_grad(v_params,
                                                  self._h_parameters,
                                                   model_transitions)
            model_v_opt_state = self._v_opt_update(self.episode, model_gradients,
                                                  lax.stop_gradient(self._v_opt_state))
            model_value_params = self._v_get_params(model_v_opt_state)

            after_real_v_tmn = self._v_network(real_value_params, h_tmn)
            after_real_v_t = self._v_network(real_value_params, h_t)

            after_model_v_tmn = self._v_network(model_value_params, h_tmn)
            after_model_v_t = self._v_network(model_value_params, h_t)

            value_loss_tmn = jnp.sum(jax.vmap(rlax.l2_loss)(after_model_v_tmn,
                                                    lax.stop_gradient(after_real_v_tmn)))

            value_loss_t = jnp.sum(jax.vmap(rlax.l2_loss)(after_model_v_t,
                                                            lax.stop_gradient(after_real_v_t)))

            # r_loss = jnp.sum(jax.vmap(rlax.l2_loss)(model_r_tmn_2_t, real_r_tmn_2_t))
            total_loss = value_loss_tmn
                         # + r_loss
                         # + value_loss_t

            return total_loss, {"corr_loss": value_loss_tmn, # + value_loss_t,
                                # "r_loss": r_loss,
                                "after_real_v_tmn": after_real_v_tmn,
                                "after_model_v_tmn": after_model_v_tmn,
                                }

        def v_planning_loss(v_params, h_params, o_params, r_params, o_t, d_t):
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            model_tmn = lax.stop_gradient(self._o_network(o_params, h_t))

            v_tmn = self._v_network(v_params, model_tmn)
            r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            r_tmn = self._r_forward(r_params, lax.stop_gradient(r_input))

            v_t_target = self._v_network(v_params, h_t)
            # to the encoded current state and the value from the predecessor latent state
            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn, d_t * jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        dwrt = [0, 1] if self._latent else 0
        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, dwrt))

        dwrt = [0, 1] if self._latent else 0
        self._latent_v_loss_grad = jax.jit(jax.value_and_grad(latent_v_loss, dwrt))

        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model, self._exploration_decay_period, 0, 1)
        # self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [1, 2, 3, 4], has_aux=True))
        self._model_loss_grad = jax.value_and_grad(model_loss, [1, 2, 3], has_aux=True)
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)

        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._lr_model)
        self._model_opt_update = jax.jit(model_opt_update)
        model_params = [self._h_parameters, self._o_parameters, self._r_parameters]
        self._model_opt_state = model_opt_init(model_params)
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
            (total_loss, losses), gradients = self._model_loss_grad(
                                                   # self._target_v_parameters,
                                                   self._v_parameters,
                                                   self._h_parameters,
                                                   self._o_parameters,
                                                   self._r_parameters,
                                                   self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                   self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._h_parameters, self._o_parameters, self._r_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                # "loss_corr": losses["corr_loss"],
                "after_real_v_tmn": losses["after_real_v_tmn"][0],
                "after_model_v_tmn": losses["after_model_v_tmn"][0],
                # "loss_r": losses["r_loss"],
                "loss_total": total_loss,
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
        o_t = np.array([timestep.observation])
        d_t = np.array([timestep.discount])
        # plan on batch of transitions

        loss, gradients = self._v_planning_loss_grad(self._v_parameters,
                                                    self._h_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    o_t, d_t)
        if self._latent:
            gradients = list(gradients)
        self._v_opt_state = self._v_opt_update(self.episode, gradients,
                                               self._v_opt_state)
        value_params = self._v_get_params(self._v_opt_state)
        if self._latent:
            self._v_parameters, _ = value_params
        else:
            self._v_parameters = value_params

        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
                                       },}
        self._log_summaries(losses_and_grads, "value_planning")

    # def _update_model_targets(self):
    #     # Periodically update the target network parameters.
    #     self._target_h_parameters, self._target_o_parameters,\
    #     self._target_r_parameters  = lax.cond(
    #         pred=jnp.mod(self.episode, self._target_update_period) == 0,
    #         true_operand=None,
    #         true_fun=lambda _: (self._h_parameters, self._o_parameters, self._r_parameters),
    #         false_operand=None,
    #         false_fun=lambda _: (self._target_h_parameters, self._target_o_parameters, self._target_r_parameters)
    #     )
    #
    # def _update_v_targets(self):
    #     # Periodically update the target network parameters.
    #     self._target_v_parameters = lax.cond(
    #         pred=jnp.mod(self.total_steps, self._target_update_period) == 0,
    #         true_operand=None,
    #         true_fun=lambda _: self._v_parameters,
    #         false_operand=None,
    #         false_fun=lambda _: self._target_v_parameters)

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def load_model(self):
        return
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_piuckle=True)[()]
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
        self._sequence.append([np.array([timestep.observation]),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array([new_timestep.observation])])
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

    def get_value_for_state(self, state, ls=None):
        features = self._get_features(state[None, ...]) if self._feature_mapper is not None else state[None, ...]
        return self._v_forward(self._v_parameters, features)[0]

