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

from prediction_agents.linear.extrinsic.lp_vanilla import LpVanilla

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class LpImplicitGen(LpVanilla):
    def __init__(
            self,
            **kwargs
    ):
        super(LpImplicitGen, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        def model_loss(o_online_params,
                       transitions):
            o_tmn_target = jnp.zeros_like(transitions[0][0])
            for i, t in enumerate(transitions[::-1]):
                o_tmn_target += (self._discount ** (i + 1)) * t[0]
            o_t = transitions[-1][-1]
            model_o_tmn = self._o_network(o_online_params, o_t)

            o_error = model_o_tmn - lax.stop_gradient(o_tmn_target)
            o_error = jnp.mean(o_error ** 2)

            return o_error

        self._o_loss_grad = jax.jit(jax.value_and_grad(model_loss, 0))
        self._o_forward = jax.jit(self._o_network)

        self._step_schedule = optimizers.inverse_time_decay(self._lr_model,
                                                            self._exploration_decay_period
                                                            * 17,
                                                            27.0)
        o_opt_init, o_opt_update, o_get_params = optimizers.adam(step_size=self._step_schedule)
        # o_opt_init, o_opt_update, o_get_params = optimizers.adam(step_size=self._lr_model)
        self._o_opt_update = jax.jit(o_opt_update)
        self._o_opt_state = o_opt_init(self._o_parameters)
        self._o_get_params = o_get_params

        def td_error(v_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            v_tm1 = self._v_network(v_params, o_tm1)
            v_t = self._v_network(v_params, o_t)
            v_target = r_t + d_t * self._discount * v_t
            td_error = lax.stop_gradient(v_tm1) - lax.stop_gradient(v_target)

            return td_error

        self._td_error = jax.jit(td_error)

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        return np.argmax(self._pi[np.argmax(timestep.observation)])

    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        transitions = [np.array([timestep.observation]),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array([new_timestep.observation])]

        loss, gradient = self._v_loss_grad(self._v_parameters,
                                transitions)
        self._v_opt_state = self._v_opt_update(self.total_steps, gradient,
                                               self._v_opt_state)
        self._v_parameters = self._v_get_params(self._v_opt_state)

        o_tmnm1 = self._o_forward(self._o_parameters, transitions[0])
        normalized_expected_o_tmn = (o_tmnm1 -
                                     np.min(o_tmnm1, axis=-1, keepdims=True)) / \
                                    (np.max(o_tmnm1, axis=-1, keepdims=True) -
                                     np.min(o_tmnm1, axis=-1, keepdims=True))
        divisior = np.sum(normalized_expected_o_tmn, axis=-1, keepdims=True)
        prob_o_tmnm = np.divide(normalized_expected_o_tmn, divisior,
                                out=np.zeros_like(o_tmnm1),
                                where=np.all(divisior != 0))

        sampled_o_tmn = np.array(
            [[np.eye(np.prod(self._input_dim))[x]
              for x in self._nrng.choice(a=range(np.prod(self._input_dim)),
                                         size=self._planning_iter,
                                         p=p_o_tmn)]
             for d, p_o_tmn in zip(divisior, prob_o_tmnm)
             if d != 0])
        sampled_o_tmn = np.reshape(sampled_o_tmn, (-1, np.prod(self._input_dim)))
        td_error = self._td_error(self._v_parameters, transitions)
        td_error = np.tile(td_error[..., None], (1, self._planning_iter))
        if len(sampled_o_tmn) > 0:
            y, vjp_fun = jax.vjp(self._v_network, self._v_parameters, sampled_o_tmn)
            v_gradient = vjp_fun(2 * td_error)[0]
            self._v_opt_state = self._v_opt_update(self.total_steps, v_gradient,
                                                   self._v_opt_state)
            self._v_parameters = self._v_get_params(self._v_opt_state)

        losses_and_grads = {"losses": {"loss_v": np.array(loss)},
                            "gradients": {"grad_norm_q":
                                              np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2)
                                                             for g in gradient]))}}
        self._log_summaries(losses_and_grads, "value")

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
            loss, gradient = self._o_loss_grad(self._o_parameters,
                                                  self._sequence)
            self._o_opt_state = self._o_opt_update(self.total_steps, gradient,
                                                           self._o_opt_state)
            self._o_parameters = self._o_get_params(self._o_opt_state)

            losses_and_grads = {"losses": {
                "loss": loss,
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
        pass

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
