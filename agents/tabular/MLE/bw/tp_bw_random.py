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


class TpRandomBw(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpRandomBw, self).__init__(**kwargs)
        self._o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        self._r_network = self._network["model"]["net"][2]

        self._sequence = []
        self._should_reset_sequence = False
        self._episode_end = False
        self._updates = np.zeros_like(self._v_network)

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

        o_t = np.array(timestep.observation)
        d_t = np.array(timestep.discount)
        losses = 0
        o_tmn = self._softmax(self._o_network[o_t])
        for prev_o_tmn in range(np.prod(self._input_dim)):
            loss, gradient = self._v_planning_loss_grad(self._v_network,
                                                           self._r_network,
                                                           o_t, prev_o_tmn, d_t)
            losses += loss
            # self._updates[prev_o_tmn] += o_tmn[prev_o_tmn] * gradient
            self._v_network[prev_o_tmn] = self._v_planning_opt_update(
                o_tmn[prev_o_tmn] * gradient,
                self._v_network[prev_o_tmn])

        # if timestep.last():
        #     self._v_network = self._v_planning_opt_update(
        #         self._updates,
        #         self._v_network)
        #     self._updates = np.zeros_like(self._v_network)
        losses_and_grads = {"losses": {"loss_v_planning": np.array(loss)},
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
        self._lr = self._initial_lr * ((total_episodes - episode) / total_episodes)
        self._lr_planning = self._initial_lr_planning * ((total_episodes - episode) / total_episodes)