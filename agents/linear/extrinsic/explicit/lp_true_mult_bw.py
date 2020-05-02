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
from basis.feature_mapper import FeatureMapper

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class LpTrueMultBw(LpVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpTrueMultBw, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        self._b_network = self._network["model"]["net"][0]
        self._a_network = self._network["model"]["net"][1]
        self._c_network = self._network["model"]["net"][2]

        self._b_parameters = self._network["model"]["params"][0]
        self._a_parameters = self._network["model"]["params"][1]
        self._c_parameters = self._network["model"]["params"][2]

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
            b_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_expected_o_tmn, o_tmn_target))
            a_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_cross_o_tmn, cross_o_tmn_target))

            model_vector_r_tmn = self._c_network(c_params,
                                                 (lax.stop_gradient(model_cross_o_tmn),
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
                               # "distance_from_wrong_cross": distance_from_wrong_cross
                               }

        def v_planning_grad(v_params, a_params, b_params, c_params, o_t,
                            d_t):
            cross_o_tmn = self._a_forward(a_params, o_t)
            expected_o_tmn = self._b_forward(b_params, o_t)
            # wrong_cross_o_tmn = lax.batch_matmul(expected_o_tmn[..., None],
            #                                jnp.transpose(expected_o_tmn[..., None], axes=[0, 2, 1]))

            vector_r_tmn = self._c_network(c_params,
                                           (cross_o_tmn, o_t))[..., None]

            v_error = d_t * jnp.array([self._discount ** self._n]) * \
                      lax.batch_matmul(expected_o_tmn[..., None],
                                       jnp.transpose(o_t[..., None], axes=[0, 2, 1]))\
                      - cross_o_tmn

            v_model_error = lax.batch_matmul(v_error, v_params[None, ])
            v_grad = jnp.mean(2 * (vector_r_tmn + v_model_error), axis=0)

            return -v_grad

        self._v_planning_loss_grad = jax.jit(v_planning_grad)
        # self._v_planning_loss_grad = v_planning_grad

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

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            (total_loss, losses), gradients = self._model_loss_grad(self._a_parameters,
                                                   self._b_parameters,
                                                   self._c_parameters,
                                                  self._sequence)
            self._model_opt_state = self._model_opt_update(self.episode, list(gradients),
                                                           self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)
            self._a_parameters, self._b_parameters, self._c_parameters = self._model_parameters

            losses_and_grads = {"losses": {
                "loss_total": total_loss,
                "loss_cross(A)": losses["cross_loss(A)"],
                "loss_o": losses["expected_loss(b)"],
                "loss_r": losses["r_loss(w_r)"],
                # "loss_distance_A": losses["distance_from_wrong_cross"],
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
        # prev_features = self._get_features([prev_timestep.observation])
        # prev_reward = self._get_features([timestep.reward])
        o_t = np.array(features)
        # o_tmn = np.array(prev_features)
        # r_tmn = np.array(prev_reward)
        d_t = np.array(timestep.discount)

        # plan on batch of transitions
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
