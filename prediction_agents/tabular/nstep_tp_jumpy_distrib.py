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


class nStepTpJumpyDistrib(VanillaTabularPrediction):
    def __init__(
            self,
            **kwargs
    ):

        super(nStepTpJumpyDistrib, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False


        def model_loss(o_params, r_params, transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            o_tmn = o_params[o_t]
            o_target = np.eye(np.prod(self._input_dim))[o_tmn_target] - o_tmn
            o_error = o_target - o_tmn
            o_loss = np.mean(o_error ** 2)

            if self._double_input_reward_model:
                r_tmn = r_params[o_tmn_target][o_t]
            else:
                r_tmn = r_params[o_tmn_target]
            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]

            r_error = (r_tmn_target - r_tmn)
            r_loss = np.mean(r_error ** 2)

            total_error = o_loss + r_loss
            return (total_error, o_loss, r_loss), (o_error, r_error)

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def v_planning_loss(v_params, o_params, r_params, o):
            o_tmn = o_params[o]
            td_errors = []
            losses = []

            divisior = np.sum(o_tmn, axis=-1, keepdims=True)
            o_tmn = np.divide(o_tmn, divisior, out=np.zeros_like(o_tmn), where=np.all(divisior != 0))
            for prev_o_tmn in range(np.prod(self._input_dim)):
                v_tmn = v_params[prev_o_tmn]
                if self._double_input_reward_model:
                    r_tmn = r_params[prev_o_tmn][o]
                else:
                    r_tmn = r_params[prev_o_tmn]
                td_error = (r_tmn + (self._discount ** self._n) *
                            v_params[o] - v_tmn)

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
            losses, gradients = self._model_loss_grad(self._o_network, self._r_network, self._sequence)
            if self._double_input_reward_model:
                self._o_network[o_t], self._r_network[o_tmn][o_t] = \
                    self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._r_network[o_tmn][o_t]])
            else:
                self._o_network[o_t], self._r_network[o_tmn] = \
                    self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._r_network[o_tmn]])
            total_loss, o_loss, r_loss = losses
            o_grad, r_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": o_loss,
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
        losses, gradients = self._v_planning_loss_grad(self._v_network,
                                                    self._o_network,
                                                    self._r_network,
                                                    o_tm1)
        o_tmnm1 = self._o_network[o_tm1]
        divisior = np.sum(o_tmnm1, axis=-1, keepdims=True)
        o_tmnm1 = np.divide(o_tmnm1, divisior, out=np.zeros_like(o_tmnm1), where=np.all(divisior != 0))
        for prev_o_tmn in range(np.prod(self._input_dim)):
            self._v_network[prev_o_tmn] = self._v_planning_opt_update(gradients[prev_o_tmn] *
                                                             o_tmnm1[:, prev_o_tmn],
                                                             self._v_network[prev_o_tmn])

        losses_and_grads = {"losses": {"loss_q_planning": np.array(np.sum(losses))},
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
        # pass
        # self._lr_model = self._initial_lr_model / (1 + episode /
        #                                            total_episodes * 0.96)
        # decay_rate = 0.1
        # self._lr_planning = self._initial_l_lr_planning / (1 + episode /
        #                                            total_episodes * decay_rate)
        # self._lr_planning = self._initial_lr_planning * \
        #                      (decay_rate ** (episode / total_episodes))
        warmup_episodes = total_episodes//3
        flat_period = total_episodes//3
        decay_period = total_episodes - warmup_episodes - flat_period
        if episode > warmup_episodes:
            steps_left = total_episodes - episode - flat_period
            if steps_left <= 0:
                return

        # step_decay =
        # step_decay = np.clip(step_decay, 0.,  self._lr_planning)
            self._lr_planning = self._initial_lr_planning * (steps_left / decay_period)
            # self._lr_model = self._initial_lr_model * (steps_left / decay_period)
        # bonus = np.clip(bonus, 0., 1. - epsilon)
        # return epsilon + bonus

