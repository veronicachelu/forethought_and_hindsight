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
    v_t,
    ):
    target_tm1 = r_t + discount_t * v_t
    return jax.lax.stop_gradient(target_tm1) - v_tm1

class LpExplicitValueBased(LpIntrinsicVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpExplicitValueBased, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        def model_loss(v_params,
                       h_params,
                       o_params,
                       r_params,
                       d_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            h_tmn = o_tmn_target if self._no_latent else self._h_network(h_params, o_tmn_target)
            h_t = o_t if self._no_latent else self._h_network(h_params, o_t)

            real_v_tmn, vjp_fun = jax.vjp(self._v_network, v_params, h_tmn)
            real_r_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_r_tmn_2_t += (self._discount ** i) * t[2]

            real_d_tmn_2_t = 0
            for i, t in enumerate(transitions):
                real_d_tmn_2_t += (self._discount ** i) * t[3]

            v_t_target = self._v_network(v_params, h_t)
            real_td_error = jax.vmap(rlax.td_learning)(real_v_tmn, real_r_tmn_2_t,
                                                       real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            real_update = vjp_fun(2 * real_td_error)[0]

            ###########################################

            model_tmn = self._o_network(o_params, h_t)
            model_v_tmn, model_vjp_fun = jax.vjp(self._v_network, v_params, model_tmn)
            if self._double_input_reward_model:
                model_r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            else:
                model_r_input = model_tmn
            model_r_tmn_2_t = self._r_network(r_params, model_r_input)

            model_td_error = jax.vmap(td_learning)(model_v_tmn, model_r_tmn_2_t,
                                                   real_d_tmn_2_t * jnp.array([self._discount ** self._n]),
                                                        v_t_target)
            model_update = model_vjp_fun(2 * model_td_error)[0]
            # corr_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_td_error, real_td_error))
            corr_loss = 0
            for i, (layer_model, layer_real) in enumerate(zip(model_update, real_update)):
                for j, (param_grad_model, param_grad_real) in enumerate(zip(layer_model, layer_real)):
                    corr_loss += jnp.mean(jax.vmap(rlax.l2_loss)(param_grad_model, lax.stop_gradient(param_grad_real)))

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn_2_t, real_r_tmn_2_t))

            total_loss = corr_loss + r_loss#+ d_loss #
            return total_loss, {"corr_loss": corr_loss,
                                "d_loss": corr_loss,
                                "r_loss": r_loss}

        def v_planning_loss(v_params, h_params, o_params, r_params, d_params, o_t, d_t):
            h_t = o_t if self._no_latent else lax.stop_gradient(self._h_network(h_params, o_t))
            model_tmn = self._o_network(o_params, h_t)

            v_tmn = self._v_network(v_params, model_tmn)
            if self._double_input_reward_model:
                r_input = jnp.concatenate([model_tmn, h_t], axis=-1)
            else:
                r_input = model_tmn
            r_tmn = self._r_forward(r_params, r_input)

            v_t_target = self._v_network(v_params, h_t)
            # to the encoded current state and the value from the predecessor latent state
            td_error = jax.vmap(rlax.td_learning)(v_tmn, r_tmn, d_t * jnp.array([self._discount ** self._n]), v_t_target)
            return jnp.mean(td_error ** 2)

        # self._v_step_schedule = optimizers.polynomial_decay(self._lr_planning, self._exploration_decay_period, 0, 2)
        # self._model_step_schedule = optimizers.polynomial_decay(self._lr_model, self._exploration_decay_period, 0, 2)

        self._v_planning_loss_grad = jax.jit(jax.value_and_grad(v_planning_loss, 0))
        # self._v_planning_loss_grad = jax.value_and_grad(v_planning_loss, 0)

        self._v_step_schedule = self._lr_planning
        self._model_step_schedule = self._lr_model

        v_opt_init, v_opt_update, v_get_params = optimizers.adam(step_size=self._v_step_schedule)
        self._v_opt_update = jax.jit(v_opt_update)
        self._v_opt_state = v_opt_init(self._v_parameters)
        self._v_get_params = v_get_params

        # This function computes dL/dTheta
        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss, [1, 2, 3, 4], has_aux=True))
        # self._model_loss_grad = jax.value_and_grad(model_loss, [1, 2, 3, 4], has_aux=True)
        self._h_forward = jax.jit(self._h_network)
        self._o_forward = jax.jit(self._o_network)
        self._r_forward = jax.jit(self._r_network)
        self._d_forward = jax.jit(self._d_network)

        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._model_step_schedule)
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
            (total_loss, losses), gradients = self._model_loss_grad(self._v_parameters,
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

    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if self._n == 0:
            return
        o_t = np.array([timestep.observation])
        d_t = np.array([timestep.discount])
        # plan on batch of transitions

        loss, gradients = self._v_planning_loss_grad(self._v_parameters,
                                                    self._h_parameters,
                                                    self._o_parameters,
                                                    self._r_parameters,
                                                    self._d_parameters,
                                                    o_t, d_t)
        self._v_opt_state = self._v_opt_update(self.episode, gradients,
                                               self._v_opt_state)
        self._v_parameters = self._v_get_params(self._v_opt_state)

        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
                                       },}
                            # "gradients": {"grad_norm_v_planning": np.sum(
                            #     np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        # return True
        return False

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

