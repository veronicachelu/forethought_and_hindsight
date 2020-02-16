import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular_agents.vanilla_tabular_agent import VanillaTabularAgent
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class DynaTabularAgent(VanillaTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(DynaTabularAgent, self).__init__(**kwargs)

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True) + 1e-8

        def model_loss(transitions):
            c_r = 1
            c_o = 1
            c_d = 1
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            o_t = self._model_network[o_tm1, a_tm1, :-3]
            r_t = self._model_network[o_tm1, a_tm1, -3]
            d_t_logits = self._model_network[o_tm1, a_tm1, -2:]
            d_t_probs = softmax(d_t_logits)

            o_error = c_o * (np.eye(np.prod(self._input_dim))[o_t_target] - o_t)
            r_error = c_r * (r_t_target - r_t)
            d_t_target = np.array(d_t_target, dtype=np.int32)
            nll = np.take_along_axis(d_t_logits, np.expand_dims(d_t_target, axis=-1), axis=1)
            d_error = - c_d * np.mean(nll)
            d_t_probs[np.arange(len(d_t_probs)), d_t_target] -= 1
            d_t_probs /= len(d_t_probs)
            d_t_probs *= c_d
            d_t_probs = -d_t_probs
            o_loss = np.mean(o_error ** 2)
            r_loss = np.mean(r_error ** 2)
            total_error = o_loss + r_loss + d_error
            return (total_error, o_loss, r_loss, d_error), (o_error, r_error, d_t_probs)

        # This function computes dL/dTheta
        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def q_planning_loss(transitions):
            o_tm1, a_tm1 = transitions
            o_t = self._model_network[o_tm1, a_tm1, :-3]
            r_t = self._model_network[o_tm1, a_tm1, -3]
            d_t = np.argmax(self._model_network[o_tm1, a_tm1, -2:], axis=-1)

            q_tm1 = self._q_network[o_tm1, a_tm1]

            q_target = r_t
            divisior = np.sum(o_t, axis=-1, keepdims=True)
            o_t = np.divide(o_t, divisior, out=np.zeros_like(o_t), where=np.all(divisior != 0))
            for next_o_t in range(np.prod(self._input_dim)):
                q_t = self._q_network[next_o_t]
                q_target += d_t * self._discount * o_t[:, next_o_t] * np.max(q_t, axis=-1)
            td_error = q_target - q_tm1

            return np.mean(td_error ** 2), td_error

        self._q_planning_loss_grad = q_planning_loss


    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        o_tm1 = np.array([timestep.observation])
        a_tm1 = np.array([action])
        r_t = np.array([new_timestep.reward])
        d_t = np.array([new_timestep.discount])
        o_t = np.array([new_timestep.observation])
        transitions = [o_tm1, a_tm1, r_t, d_t, o_t]

        losses, gradients = self._model_loss_grad(transitions)
        self._model_network[o_tm1, a_tm1, :-3], \
        self._model_network[o_tm1, a_tm1, -3], \
        self._model_network[o_tm1, a_tm1, -2:] = self._model_opt_update(gradients,
                                                                        [self._model_network[o_tm1, a_tm1, :-3],
                                                                         self._model_network[o_tm1, a_tm1, -3], \
                                                                         self._model_network[o_tm1, a_tm1, -2:]
                                                                        ])
        total_loss, o_loss, r_loss, d_loss = losses
        o_grad, r_grad, d_grad = gradients
        o_grad = np.sum(np.linalg.norm(np.asarray(o_grad), ord=2), axis=-1)
        r_grad = np.sum(np.linalg.norm(np.asarray(r_grad), ord=2), axis=-1)
        d_grad = np.sum(np.linalg.norm(np.asarray(d_grad), ord=2), axis=-1)
        losses_and_grads = {"losses": {
            "loss": total_loss,
            "o_loss": o_loss,
            "r_loss": r_loss,
            "d_loss": d_loss,
            "o_grad": o_grad,
            "r_grad": r_grad,
            "d_grad": d_grad
        },
        }
        self._log_summaries(losses_and_grads, "model")

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return

        for k in range(self._planning_iter):
            transitions = self._replay.sample(self._batch_size)
            # plan on batch of transitions
            o_tm1, a_tm1 = transitions

            loss, gradient = self._q_planning_loss_grad(transitions)
            self._q_network[o_tm1, a_tm1] = self._q_opt_update(gradient, self._q_network[o_tm1, a_tm1])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                }
            self._log_summaries(losses_and_grads, "value_planning")

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        self._replay.add([
            timestep.observation,
            action,
        ])

    def load_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._q_network = to_load["q_parameters"]
            self._replay = to_load["replay"]
            self._model_network = to_load["model_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "q_parameters": self._q_network,
            "replay": self._replay,
            "model_parameters": self._model_network,
        }
        np.save(checkpoint, to_save)
        print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def _log_summaries(self, losses_and_grads, summary_name):
        losses = losses_and_grads["losses"]

        if self.episode % self._log_period == 0:
            for k, v in losses.items():
                tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), losses[k], step=self.episode)
            self.writer.flush()

