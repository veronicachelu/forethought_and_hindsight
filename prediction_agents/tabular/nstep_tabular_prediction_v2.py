import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from prediction_agents.tabular.vanilla_tabular_prediction import VanillaTabularPrediction
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class nStepTabularPredictionV2(VanillaTabularPrediction):
    def __init__(
            self,
            **kwargs
    ):

        super(nStepTabularPredictionV2, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True) + 1e-8

        self._softmax = softmax

        def sigmoid(x):
            """Compute sigmoid values for each sets of scores in x."""
            "Numerically stable sigmoid function."
            if x >= 0:
                z = np.exp(-x)
                return 1 / (1 + z)
            else:
                # if x is less than zero then z will be small, denom can't be
                # zero because it's 1+z.
                z = np.exp(x)
                return z / (1 + z)

        self._sigmoid = sigmoid

        def model_loss(o_params, r_params, d_params, transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]
            for i, t in enumerate(np.flip(transitions, axis=0)):
                if t[3] == 0:
                    o_tmn_target = transitions[i][0]
                    break

            o_tmn = o_params[o_t]
            o_error = np.eye(np.prod(self._input_dim))[o_tmn_target] - o_tmn
            o_loss = np.mean(o_error ** 2)

            r_tmn = r_params[o_tmn_target]
            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]
                if t[3] == 0:
                    break

            r_error = r_tmn_target - r_tmn
            r_loss = np.mean(r_error ** 2)

            d_tmn_logit = d_params[o_tmn_target]
            d_tmn_target = 1
            for i, t in enumerate(transitions):
                d_tmn_target *= self._discount * t[3]
                # if t[3] == 0:
                #     break
            d_tmn_prob = sigmoid(d_tmn_logit)
            # d_error = - 10 * d_tmn_prob * (1 - d_tmn_prob)
            d_error = - (d_tmn_prob - d_tmn_target)
            d_loss = -(np.maximum(d_tmn_logit, 0) - d_tmn_logit * d_tmn_target + np.log(1 + np.exp(-np.abs(d_tmn_logit))))
            # d_error = d_tmn_target - d_tmn
            # d_loss = np.mean(d_error ** 2)

            # d_tmn_logits = d_params[o_tmn_target]
            # d_tmn_target = np.all([bool(t[3]) for t in transitions])
            # d_tmn_probs = softmax(d_tmn_logits)
            # d_tmn_target = np.array(d_tmn_target, dtype=np.int32)
            # # nll = np.take_along_axis(d_tmn_logits, np.expand_dims(d_tmn_target, axis=-1), axis=1)
            # nll = d_tmn_logits[d_tmn_target]
            # # d_error = - np.mean(nll)
            # d_error = - nll
            # # d_tmn_probs[np.arange(len(d_tmn_probs)), d_tmn_target] -= 1
            # d_tmn_probs[d_tmn_target] -= 1
            # # d_tmn_probs /= len(d_tmn_probs)
            # d_tmn_probs = -d_tmn_probs

            total_error = o_loss + r_loss + d_loss
            return (total_error, o_loss, r_loss, d_loss), (o_error, r_error, d_error)

        # This function computes dL/dTheta
        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def v_planning_loss(v_params, o_params, r_params, d_params, o):
            o_tmn = o_params[o]
            td_errors = []
            losses = []

            divisior = np.sum(o_tmn, axis=-1, keepdims=True)
            o_tmn = np.divide(o_tmn, divisior, out=np.zeros_like(o_tmn), where=np.all(divisior != 0))
            for prev_o_tmn in range(np.prod(self._input_dim)):
                v_tmn = v_params[prev_o_tmn]
                r_tmn = r_params[prev_o_tmn]
                # d_tmn = self._sigmoid(d_params[prev_o_tmn])
                d_tmn = self._discount ** self._n
                td_error = (r_tmn + d_tmn * v_params[o] - v_tmn)

                td_errors.append(td_error)
                loss = td_error ** 2
                losses.append(o_tmn[:, prev_o_tmn] * loss)

            return losses, td_errors

        self._v_planning_loss_grad = v_planning_loss

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
            o_t = self._sequence[-1][-1]
            losses, gradients = self._model_loss_grad(self._o_network, self._r_network, self._d_network, self._sequence)
            self._o_network[o_t], self._r_network[o_tmn], self._d_network[o_tmn] = \
                self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._r_network[o_tmn],
                                                   self._d_network[o_tmn]])
            total_loss, o_loss, r_loss, d_loss = losses
            o_grad, r_grad, d_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
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
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False

    def planning_update(
            self,
            timestep: dm_env.TimeStep
    ):
        # pass
        o_tm1 = np.array([timestep.observation])
        losses, gradients = self._v_planning_loss_grad(self._v_network,
                                                    self._o_network,
                                                    self._r_network,
                                                    self._d_network,
                                                    o_tm1)
        o_tmnm1 = self._o_network[o_tm1]
        divisior = np.sum(o_tmnm1, axis=-1, keepdims=True)
        o_tmnm1 = np.divide(o_tmnm1, divisior, out=np.zeros_like(o_tmnm1), where=np.all(divisior != 0))
        for prev_o_tmn in range(np.prod(self._input_dim)):
            self._v_network[prev_o_tmn] = self._v_opt_update(gradients[prev_o_tmn] *
                                                             o_tmnm1[:, prev_o_tmn],
                                                             self._v_network[prev_o_tmn])

        losses_and_grads = {"losses": {"loss_q_planning": np.array(np.mean(losses))},
                            }
        self._log_summaries(losses_and_grads, "value_planning")

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
            self._v_network = to_load["v_parameters"]
            self._replay = to_load["replay"]
            self._o_network = to_load["o_parameters"]
            self._r_network = to_load["r_parameters"]
            self._d_network = to_load["d_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "v_parameters": self._v_network,
            "replay": self._replay,
            "o_parameters": self._o_network,
            "r_parameters": self._r_network,
            "d_parameters": self._d_network,
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

        # self._replay.add([
        #     timestep.observation,
        # ])

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

    def update_hyper_params(self, step, total_steps):
        pass

