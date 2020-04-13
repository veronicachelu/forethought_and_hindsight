import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular.tp_vanilla import TpVanilla
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpFwBwMG(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpFwBwMG, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False
        self._replay._alpha = 1.0
        self._replay._initial_beta = 1.0
        self._replay._beta = self._replay._initial_beta

        def model_loss(bw_o_params, fw_o_params, r_params, transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]

            model_o_tmn = bw_o_params[o_t]
            bw_o_target = np.eye(np.prod(self._input_dim))[o_tmn_target] - model_o_tmn

            bw_o_error = bw_o_target - model_o_tmn
            bw_o_loss = np.mean(bw_o_error ** 2)

            # forward
            model_o_t = fw_o_params[o_tmn_target]
            fw_o_target = np.eye(np.prod(self._input_dim))[o_t] - model_o_t
            fw_o_error = fw_o_target - model_o_t
            fw_o_loss = np.mean(fw_o_error ** 2)

            if self._double_input_reward_model:
                r_tmn = r_params[o_tmn_target][o_t]
            else:
                r_tmn = r_params[o_tmn_target]
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

        def v_planning_loss(v_params, fw_o_params, r_params, o_tmn):
            o_tmn = o_tmn[0]
            o_t = fw_o_params[o_tmn]
            r_tmn = r_params[o_tmn]
            v_tmn = v_params[o_tmn]

            if self._double_input_reward_model:
                target = 0
            else:
                target = r_tmn

            divisior = np.sum(o_t, axis=-1, keepdims=True)
            o_t = np.divide(o_t, divisior, out=np.zeros_like(o_t), where=np.all(divisior != 0))
            for next_o_t in range(np.prod(self._input_dim)):
                if self._double_input_reward_model:
                    target_per_next_o = o_t[next_o_t] * \
                    (r_tmn[next_o_t] + (self._discount ** self._n) *\
                          v_params[next_o_t])
                else:
                    target_per_next_o = o_t[next_o_t] * \
                          (self._discount ** self._n) *\
                          v_params[next_o_t]
                target += target_per_next_o
            td_error = (target - v_tmn)
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
            losses, gradients = self._model_loss_grad(self._o_network, self._fw_o_network, self._r_network, self._sequence)
            if self._double_input_reward_model:
                self._o_network[o_t], \
                self._fw_o_network[o_tmn], \
                self._r_network[o_tmn][o_t] = \
                    self._model_opt_update(gradients, [self._o_network[o_t],
                                                       self._fw_o_network[o_tmn],
                                                       self._r_network[o_tmn][o_t]])
            else:
                self._o_network[o_t], \
                self._fw_o_network[o_tmn], \
                self._r_network[o_tmn] = \
                    self._model_opt_update(gradients, [self._o_network[o_t],
                                                       self._fw_o_network[o_tmn],
                                                       self._r_network[o_tmn]])
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
        if self._replay.size < self._min_replay_size:
            return
        if self.total_steps % self._planning_period == 0:
            for k in range(self._planning_iter):
                weights, priority_transitions = self._replay.sample_priority(1)
                priority = priority_transitions[0][0]
                o_tm1 = priority_transitions[1][0]

                o_tmnm1 = self._o_network[o_tm1]
                divisior = np.sum(o_tmnm1, axis=-1, keepdims=True)
                o_tmnm1 = np.divide(o_tmnm1, divisior, out=np.zeros_like(o_tmnm1), where=np.all(divisior != 0))

                for prev_o_tmnm1 in range(np.prod(self._input_dim)):
                    if o_tmnm1[:, prev_o_tmnm1] == 0:
                        continue
                    from_o = [prev_o_tmnm1]
                    loss, gradient = self._v_planning_loss_grad(self._v_network,
                                                                self._fw_o_network,
                                                                self._r_network,
                                                                from_o)
                    self._v_network[from_o] = self._v_planning_opt_update(gradient,
                                                                     self._v_network[from_o])


                    losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                        }
                    self._log_summaries(losses_and_grads, "value_planning")

                    self._replay.add([
                        np.abs(gradient),
                        np.array(from_o),
                    ])

    def value_update(
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

        loss, gradient = self._v_loss_grad(self._v_network, transitions)
        self._v_network[o_tm1] = self._v_opt_update(gradient, self._v_network[o_tm1])

        losses_and_grads = {"losses": {"loss_v": np.array(loss)},
                            }
        self._log_summaries(losses_and_grads, "value")

        self._replay.add([
            np.abs(gradient),
            o_tm1,
        ])


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
        warmup_episodes = 0
        flat_period = 0
        decay_period = total_episodes - warmup_episodes - flat_period
        if episode > warmup_episodes:
            steps_left = total_episodes - episode - flat_period
            if steps_left <= 0:
                return

            self._lr_planning = self._initial_lr_planning * (steps_left / decay_period)


