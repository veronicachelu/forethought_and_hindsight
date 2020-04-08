import os
from typing import Any
from typing import Callable, Sequence

import dm_env
import numpy as np
import tensorflow as tf
from jax import numpy as jnp

from action_prediction_agents.tabular.tp_qvanilla import TpQVanilla

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpQExplicitDistrib(TpQVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpQExplicitDistrib, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True) + 1e-8

        self._softmax = softmax

        def model_loss(o_params, a_params, r_params, transitions):
            o_tmn_target = transitions[0][0]
            a_tmn_target = transitions[0][1]
            a_tmn_target = np.array(a_tmn_target, dtype=np.int32)
            o_t = transitions[-1][-1]

            o_tmn = o_params[o_t]

            o_target = np.eye(np.prod(self._input_dim))[o_tmn_target] - o_tmn
            o_error = o_target - o_tmn
            o_loss = np.mean(o_error ** 2)

            a_tmn_logits = a_params[o_tmn_target][o_t]
            a_tmn_probs = self._softmax(a_tmn_logits)
            nll = np.take_along_axis(a_tmn_logits,
                                     np.expand_dims(a_tmn_probs, axis=-1),
                                     axis=-1)
            a_loss = -np.mean(nll)

            a_tmn_probs[np.arange(len(a_tmn_probs)), a_tmn_target] -= 1
            a_tmn_probs /= len(a_tmn_probs)
            a_error = -a_tmn_probs

            r_tmn = r_params[o_tmn_target][o_t][a_tmn_target]

            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]

            r_error = r_tmn_target - r_tmn
            r_loss = np.mean(r_error ** 2)

            total_error = o_loss + r_loss + a_loss
            return (total_error, o_loss, a_loss, r_loss), (o_error, a_error, r_error)

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def q_planning_loss(q_params, o_params, a_params, r_params, o):
            o_tmn = o_params[o]
            td_errors = []
            losses = []

            divisior = np.sum(o_tmn, axis=-1, keepdims=True)
            o_tmn = np.divide(o_tmn, divisior, out=np.zeros_like(o_tmn), where=np.all(divisior != 0))
            for prev_o_tmn in range(np.prod(self._input_dim)):
                a_tmn_probs = np.softmax(a_params[o_tmn][o], -1)
                for prev_a_tmn in range(self._nA):
                    q_tmn = q_params[prev_o_tmn][prev_a_tmn]
                    r_tmn = r_params[prev_o_tmn][o][prev_a_tmn]

                    td_error = (r_tmn + (self._discount ** self._n) *
                                q_params[o] - q_tmn)

                    td_errors.append(o_tmn[:, prev_o_tmn] * a_tmn_probs[:,prev_o_tmn] * td_error)
                    loss = td_error ** 2
                    losses.append(o_tmn[:, prev_o_tmn] * loss)

            return losses, td_errors

        self._q_planning_loss_grad = q_planning_loss

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            o_tmn = self._sequence[0][0]
            a_tmn = self._sequence[0][1]
            o_t = self._sequence[-1][-1]
            losses, gradients = self._model_loss_grad(self._o_network, self._r_network, self._sequence)
            self._o_network[o_t], self._a_network[o_tmn][o_t], self._r_network[o_tmn][o_t][a_tmn] = \
                self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._a_network[o_tmn][o_t],
                                               self._r_network[o_tmn][o_t][a_tmn]])

            total_loss, o_loss, a_loss, r_loss = losses
            o_grad, a_grad, r_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": o_loss,
                "a_loss": a_loss,
                "r_loss": r_loss,
                "o_grad": o_grad,
                "r_grad": r_grad,
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False

    def planning_update(
            self,
            timestep: dm_env.TimeStep
    ):
        o_tm1 = np.array([timestep.observation])
        losses, gradients = self._q_planning_loss_grad(self._q_network,
                                                    self._o_network,
                                                    self._a_network,
                                                    self._r_network,
                                                    o_tm1)
        o_tmnm1 = self._o_network[o_tm1]

        # divisior = np.sum(o_tmnm1, axis=-1, keepdims=True)
        # o_tmnm1 = np.divide(o_tmnm1, divisior, out=np.zeros_like(o_tmnm1), where=np.all(divisior != 0))
        i = 0
        for prev_o_tmn in range(np.prod(self._input_dim)):
            for prev_a_tmn in range(self._nA):
                a_tmn_logits = self._a_network[prev_o_tmn][o_tm1]
                a_tmn_prob = self._softmax(a_tmn_logits)
                self._q_network[prev_o_tmn][a_tmn_prob] = self._q_planning_opt_update(gradients[i],
                                                                 self._q_network[prev_o_tmn][a_tmn_prob])
                i+= 1


        losses_and_grads = {"losses": {"loss_q_planning": np.array(np.sum(losses))},
                            }
        self._log_summaries(losses_and_grads, "q_planning")

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
            self._q_network = to_load["q_parameters"]
            self._o_network = to_load["o_parameters"]
            self._r_network = to_load["r_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "v_parameters": self._q_network,
            "o_parameters": self._o_network,
            "r_parameters": self._r_network,
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
        self._sequence.append([timestep.observation,
                               action,
                               new_timestep.reward,
                               new_timestep.discount,
                               new_timestep.observation])
        if new_timestep.discount == 0:
            self._should_reset_sequence = True

    def _log_summaries(self, losses_and_grads, summary_name):
        losses = losses_and_grads["losses"]

        if self._max_len == -1:
            ep = self.total_steps
        else:
            ep = self.episode
        if ep % self._log_period == 0:
            for k, v in losses.items():
                tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), losses[k], step=self.total_steps)
            self.writer.flush()

    def update_hyper_params(self, episode, total_episodes):
        warmup_episodes = 0
        flat_period = 0
        decay_period = total_episodes - warmup_episodes - flat_period
        if episode > warmup_episodes:
            steps_left = total_episodes - episode - flat_period
            if steps_left <= 0:
                return
            self._lr_planning = self._initial_lr_planning * (steps_left / decay_period)

