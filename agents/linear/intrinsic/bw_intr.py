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

class LpBwIntr(LpIntrinsicVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpBwIntr, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._target_update_period = 1

        def model_loss(v_params,
                       h_params,
                       o_params,
                       r_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            h_tmn = self._h_network(h_params, o_tmn_target) if self._latent else o_tmn_target
            h_t = self._h_network(h_params, o_t) if self._latent else o_t

            # #compute fwd + bwd pass
            real_v_tmn, vjp_fun = jax.vjp(self._v_network, v_params, h_tmn)
            real_v_tmn = jnp.squeeze(real_v_tmn, axis=-1)
            real_r_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_r_tmn_2_t += (self._discount ** i) * t[2]

            real_d_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_d_tmn_2_t += t[3]

            v_t_target = jnp.squeeze(self._v_network(v_params, h_t), axis=-1)
            real_td_error = jax.vmap(td_learning)(real_v_tmn, real_r_tmn_2_t,
                                                       real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            real_update = vjp_fun(real_td_error[..., None])[0]# pull back real_td_error

            model_tmn = self._o_network(o_params, h_t)
            model_v_tmn, model_vjp_fun = jax.vjp(self._v_network, v_params, model_tmn)
            model_v_tmn = jnp.squeeze(model_v_tmn, axis=-1)

            model_r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            model_r_tmn_2_t = self._r_network(r_params, model_r_input)

            model_td_error = jax.vmap(td_learning)(model_v_tmn, model_r_tmn_2_t,
                                                   real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            model_update = model_vjp_fun(model_td_error[None, ...])[0] # pullback model_td_error

            corr_loss = jnp.sum(jax.vmap(rlax.l2_loss)(model_update, lax.stop_gradient(real_update)))
            # r_loss = jnp.sum(jax.vmap(rlax.l2_loss)(model_r_tmn_2_t, real_r_tmn_2_t))
            total_loss = corr_loss

            return total_loss, {"corr_loss": corr_loss,}
                                # "r_loss": r_loss}

        def v_planning_loss(v_params, h_params, o_params, r_params, o_t, d_t):
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            model_tmn = lax.stop_gradient(self._o_network(o_params, h_t))
            v_tmn = jnp.squeeze(self._v_network(v_params, model_tmn), axis=-1)
            r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            r_tmn = self._r_network(r_params, lax.stop_gradient(r_input))
            v_t_target = jnp.squeeze(self._v_network(v_params, h_t), axis=-1)

            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn, d_t * jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        dwrt = [0, 1] if self._latent else 0
        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, dwrt))
        self._model_step_schedule = optimizers.polynomial_decay(self._lr_model,
                                                                self._exploration_decay_period, 0, 0.9)
        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [1, 2, 3], has_aux=True))
        # self._model_loss_grad = (jax.value_and_grad(model_loss, [1, 2, 3], has_aux=True))
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)

        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
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
        features = self._get_features([timestep.observation])
        o_t = np.array(features)
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

