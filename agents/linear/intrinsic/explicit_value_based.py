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

class LpExplicitValueBased(LpIntrinsicVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpExplicitValueBased, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._target_update_period = 50

        def model_loss(v_params,
                       h_params,
                       o_params,
                       r_params,
                       d_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            h_tmn = self._h_network(h_params, o_tmn_target) if self._latent else o_tmn_target
            h_t = self._h_network(h_params, o_t) if self._latent else o_t
            #
            # #compute fwd + bwd pass
            real_v_tmn, vjp_fun = jax.vjp(self._v_network, v_params, h_tmn)
            real_r_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_r_tmn_2_t += (self._discount ** i) * t[2]
            #
            real_d_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_d_tmn_2_t += (self._discount ** i) * t[3]

            v_t_target = self._v_network(v_params, h_t)
            real_td_error = jax.vmap(rlax.td_learning)(real_v_tmn, real_r_tmn_2_t,
                                                       real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            real_update = vjp_fun(2 * real_td_error)[0]
            # # When the
            # #  Delta_m (v) = (T_m v - v) dv
            # # is minimized, then
            # #  v* = T_m_fw v*  with v* the fixed point
            # #  (this is done by the model free process - fw looking)
            # # v* is the optimal value function wrt to the real model m
            # #  Let be a credit discriminator between the optimal model and the estim model
            # # L = |Delta_hatm(v) - Delta_m (v)|^2_2
            # # dL/dtheta =  (Delta_hatm (v) - Delta_m (v)) (dDelta_hatm (v))/dtheta
            # # If v = v*
            # # dL/dtheta =  (T_hatm v* - v*) dv* (- dT_hatm/dtheta)
            # #
            # ###########################################
            #
            model_tmn = self._o_network(o_params, h_t)
            model_v_tmn, model_vjp_fun = jax.vjp(self._v_network, v_params, model_tmn)
            model_r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            model_r_tmn_2_t = self._r_network(r_params, model_r_input)

            model_td_error = jax.vmap(rlax.td_learning)(model_v_tmn, model_r_tmn_2_t,
                                                   real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            model_update = model_vjp_fun(2 * model_td_error)[0]
            # corr_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_td_error, real_td_error))
            corr_loss = 0
            for i, (layer_model, layer_real) in enumerate(zip(model_update, real_update)):
                for j, (param_grad_model, param_grad_real) in enumerate(zip(layer_model, layer_real)):
                    corr_loss += jnp.sum(jax.vmap(rlax.l2_loss)(param_grad_model,
                                                                 lax.stop_gradient(param_grad_real)))

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn_2_t, real_r_tmn_2_t))
            total_loss = corr_loss + r_loss#+ d_loss #


            ###### Alternative
            # model_tmn = self._o_network(o_params, h_t)
            # model_v_tmn = self._v_network(v_params, model_tmn)
            # real_v_tmn = self._v_network(v_params, h_tmn)
            #
            # l_m = jnp.mean(jax.vmap(rlax.l2_loss)(model_v_tmn, lax.stop_gradient(real_v_tmn)))
            # # model_r_tmn_2_t,
            # #                                        real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
            #                                        #      v_t_target)
            # total_loss = corr_loss + r_loss
            return total_loss, {"corr_loss": corr_loss,
                                "d_loss": total_loss,
                                "r_loss": r_loss}

        def v_planning_loss(v_params, h_params, o_params, r_params, d_params, o_t):
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            model_tmn = lax.stop_gradient(self._o_network(o_params, h_t))

            v_tmn = self._target_v_network(v_params, model_tmn)
            r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            r_tmn = self._r_forward(r_params, lax.stop_gradient(r_input))

            v_t_target = self._target_v_network(v_params, h_t)
            # to the encoded current state and the value from the predecessor latent state
            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn, jnp.array([self._discount ** self._n]),
                                                  v_t_target)
            return jnp.mean(td_error ** 2)

        # self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, 0))
        # v_planning_opt_init, v_planning_opt_update, v_planning_get_params = optimizers.adam(step_size=self._lr_planning)
        # self._v_planning_opt_update = jax.jit(v_planning_opt_update)
        # self._v_planning_opt_state = v_planning_opt_init(self._target_v_parameters)
        # self._v_planning_get_params = v_planning_get_params
        # self._target_v_forward = jax.jit(self._target_v_network)

        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [1, 2, 3, 4], has_aux=True))
        # self._model_loss_grad = jax.value_and_grad(model_loss, [1, 2, 3, 4], has_aux=True)
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)
        self._d_forward = jax.jit(self._d_network)

        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._lr_model)
        self._model_opt_update = jax.jit(model_opt_update)
        model_params = [self._h_parameters, self._o_parameters, self._r_parameters, self._d_parameters]
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
                                                   self._target_v_parameters,
                                                   self._h_parameters,
                                                   self._o_parameters,
                                                   self._r_parameters,
                                                   self._d_parameters,
                                                   self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                   self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._h_parameters, self._o_parameters, self._r_parameters, self._d_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_corr": losses["corr_loss"],
                "loss_r": losses["r_loss"],
                "loss_d": losses["d_loss"],
                "loss_total": total_loss,
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False

        self._update_v_targets()
        # self._update_model_targets()

    # def planning_update(
    #         self,
    #         timestep: dm_env.TimeStep,
    #         prev_timestep=None
    # ):
    #     return
    #     if self._n == 0:
    #         return
    #     if timestep.discount is None:
    #         return
    #     o_t = np.array([timestep.observation])
    #     d_t = np.array([timestep.discount])
    #     # plan on batch of transitions
    #
    #     loss, gradients = self._v_planning_loss_grad(self._target_v_parameters,
    #                                                 self._h_parameters,
    #                                                 self._o_parameters,
    #                                                 self._r_parameters,
    #                                                 self._d_parameters,
    #                                                 o_t)
    #     self._v_planning_opt_state = self._v_planning_opt_update(self.episode, gradients,
    #                                            self._v_planning_opt_state)
    #     self._target_v_parameters = self._v_planning_get_params(self._v_planning_opt_state)
    #
    #     losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
    #                                    },}
    #                         # "gradients": {"grad_norm_v_planning": np.sum(
    #                         #     np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
    #     self._log_summaries(losses_and_grads, "value_planning")
    #
    #     # self._update_v_targets()
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

    def _update_v_targets(self):
        # Periodically update the target network parameters.
        self._target_v_parameters = lax.cond(
            pred=jnp.mod(self.total_steps, self._target_update_period) == 0,
            true_operand=None,
            true_fun=lambda _: self._v_parameters,
            false_operand=None,
            false_fun=lambda _: self._target_v_parameters)

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

    def get_values_for_all_states(self, all_states):
        latents = self._h_forward(self._h_parameters, np.array(all_states)) if self._latent else all_states
        return np.array(self._v_forward(self._v_parameters, np.asarray(latents, np.float)), np.float)


