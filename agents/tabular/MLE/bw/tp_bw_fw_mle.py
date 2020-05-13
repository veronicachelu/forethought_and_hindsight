import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular.MLE.tp_vanilla import TpVanilla
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpBwFwMLE(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpBwFwMLE, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        self._o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        self._r_network = self._network["model"]["net"][2]
        self._d_network = self._network["model"]["net"][3]

        def model_loss(bw_o_params, fw_o_params, r_params, transitions):
            o_tmn = transitions[0][0]
            o_t = transitions[-1][-1]

            bw_o_tmn = bw_o_params[o_t]
            bw_o_tmn_target = np.eye(np.prod(self._input_dim))[o_tmn]
            bw_o_loss = self._ce(self._log_softmax(bw_o_tmn), bw_o_tmn_target)
            bw_o_tmn_probs = self._softmax(bw_o_tmn)
            bw_o_tmn_probs[o_tmn] -= 1
            bw_o_tmn_probs /= len(bw_o_tmn_probs)
            bw_o_error = - bw_o_tmn_probs

            fw_o_t = fw_o_params[o_tmn]
            fw_o_t_target = np.eye(np.prod(self._input_dim))[o_t]
            fw_o_loss = self._ce(self._log_softmax(fw_o_t), fw_o_t_target)
            fw_o_t_probs = self._softmax(fw_o_t)
            fw_o_t_probs[o_t] -= 1
            fw_o_t_probs /= len(fw_o_t_probs)
            fw_o_error = - fw_o_t_probs

            r_tmn = r_params[o_tmn, o_t]
            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]
            r_error = r_tmn_target - r_tmn
            r_loss = np.mean(r_error ** 2)

            total_error = bw_o_loss + fw_o_loss + r_loss
            return (total_error, bw_o_loss, fw_o_loss, r_loss), (bw_o_error, fw_o_error, r_error)

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def v_planning_loss(v_params, bw_o_params, fw_o_params, r_params, o_t, o_tmn, d_t):
            v_tmn = v_params[o_tmn]
            r_tmn = r_params[o_tmn, o_t]

            # target_correction = 0
            # for potential_o_t in range(np.prod(self._input_dim)):
            #     if potential_o_t != o_t:
            #         target_correction += self._softmax(bw_o_params[potential_o_t])[o_tmn] * \
            #         (r_params[o_tmn, potential_o_t] + (self._discount ** self._n) * v_params[potential_o_t])
            #
            # td_error = self._softmax(bw_o_params[o_t])[o_tmn] * (r_tmn + d_t * (self._discount ** self._n) *
            #             v_params[o_t]) + target_correction - v_tmn

            td_error = self._softmax(bw_o_params[o_t])[o_tmn] * (r_tmn + d_t * (self._discount ** self._n) *
                             v_params[o_t] - v_tmn)
            target = 0
            for potential_o_t in range(np.prod(self._input_dim)):
                if potential_o_t != o_t:
                    target += self._softmax(fw_o_params[o_tmn])[potential_o_t] * \
                                         (r_params[o_tmn, potential_o_t] + (self._discount ** self._n) * v_params[
                                             potential_o_t] - v_tmn)

            # td_error += self._softmax(bw_o_params[o_t])[o_tmn] * (target - v_tmn)

            loss = td_error ** 2

            return loss, td_error

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
            losses, gradients = self._model_loss_grad(self._o_network,
                                                      self._fw_o_network,
                                                      self._r_network,
                                                      self._sequence)
            self._o_network[o_t], \
            self._fw_o_network[o_tmn], \
            self._r_network[o_tmn, o_t] = \
                self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._fw_o_network[o_tmn],
                                               self._r_network[o_tmn, o_t]])
            total_loss, bw_o_loss, fw_o_loss, r_loss = losses
            bw_o_grad, fw_o_grad, r_grad = gradients
            bw_o_grad = np.linalg.norm(np.asarray(bw_o_grad), ord=2)
            fw_o_grad = np.linalg.norm(np.asarray(fw_o_grad), ord=2)
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "bw_o_loss": bw_o_loss,
                "fw_o_loss": fw_o_loss,
                "r_loss": r_loss,
                "bw_o_grad": bw_o_grad,
                "fw_o_grad": fw_o_grad,
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
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if timestep.discount is None:
            return

        o_t = np.array(timestep.observation)
        d_t = np.array(timestep.discount)
        losses = 0

        for prev_o_tmn in range(np.prod(self._input_dim)):
            loss, gradient = self._v_planning_loss_grad(self._v_network,
                                                        self._o_network,
                                                        self._fw_o_network,
                                                           self._r_network,
                                                           o_t, prev_o_tmn, d_t)
            losses += loss
            self._v_network[prev_o_tmn] = self._v_planning_opt_update(
                gradient,
                self._v_network[prev_o_tmn])

        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss)},
                            }
        self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        # return False
        return True

    def load_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            if os.path.exists(checkpoint):
                to_load = np.load(checkpoint, allow_pickle=True)[()]
                self.episode = to_load["episode"]
                self.total_steps = to_load["total_steps"]
                self._v_network = to_load["v_parameters"]
                self._o_network = to_load["o_parameters"]
                self._r_network = to_load["r_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")

    def save_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "v_parameters": self._v_network,
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
        if self._logs is not None:
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
        self._lr = self._initial_lr * ((total_episodes - episode) / total_episodes)
        self._lr_model = self._initial_lr_model * ((total_episodes - episode) / total_episodes)
        self._lr_planning = self._initial_lr_planning * ((total_episodes - episode) / total_episodes)