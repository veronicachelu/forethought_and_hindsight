import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp
from collections import deque
from agents.tabular.MLE.tp_vanilla import TpVanilla
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpTrueBwRecur(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpTrueBwRecur, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        def v_planning_loss(v_params, r_params, o, o_tmn, d_t):
            v_tmn = v_params[o_tmn]
            r_tmn = r_params[o_tmn, o]
            td_error = (r_tmn + d_t * (self._discount ** self._n) *
                        v_params[o] - v_tmn)

            loss = td_error ** 2

            return loss, td_error

        self._v_planning_loss_grad = v_planning_loss

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        pass

    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if timestep.discount is None:
            return

        self.planning_update_bfs(timestep, prev_timestep)

    def planning_update_bfs(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        def in_queue(el):
            for qel in traj:
                if qel[0] == el:
                    return True
            else:
                return False
        traj = deque()
        traj.append((timestep.observation, timestep.discount, 0))
        sum_of_losses = 0
        while len(traj) > 0:
            o_t, d_t, recur_level = traj.pop()
            o_tmn = self._o_network[o_t]
            divisior = np.sum(o_tmn, axis=-1, keepdims=False)
            recur_level += 1
            if divisior == 0 or recur_level > 2:
                continue
            for prev_o_tmn in range(np.prod(self._input_dim)):
                if o_tmn[prev_o_tmn] != 0:
                    loss, gradient = self._v_planning_loss_grad(self._v_network,
                                                                   self._r_network,
                                                                   o_t, prev_o_tmn, d_t)
                    sum_of_losses += loss
                    self._v_network[prev_o_tmn] = self._v_planning_opt_update(o_tmn[prev_o_tmn] * gradient,
                                                                              self._v_network[prev_o_tmn])
                    if not in_queue(prev_o_tmn) and prev_o_tmn != o_t:
                        traj.append((prev_o_tmn, 1, recur_level))

        losses_and_grads = {"losses": {"loss_v_planning": np.array(sum_of_losses)},
                                    }
        self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True
        # return False

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
       pass

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
        warmup_episodes = 0
        flat_period = 0
        decay_period = total_episodes - warmup_episodes - flat_period
        if episode > warmup_episodes:
            steps_left = total_episodes - episode - flat_period
            if steps_left <= 0:
                return
            self._lr_planning = self._initial_lr_planning * (steps_left / decay_period)

