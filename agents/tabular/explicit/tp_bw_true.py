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


class TpTrueBw(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpTrueBw, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        def v_planning_loss(v_params, o_params, r_params, o, d):
            o_tmn = o_params[o]
            td_errors = []
            losses = []

            for prev_o_tmn in range(np.prod(self._input_dim)):
                v_tmn = v_params[prev_o_tmn]
                r_tmn = r_params[prev_o_tmn, o]
                td_error = (r_tmn + d * (self._discount ** self._n) *
                            v_params[o] - v_tmn)

                td_errors.append(o_tmn[:, prev_o_tmn] * td_error)
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
        pass

    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if timestep.discount is None:
            return

        o_tm1 = np.array([timestep.observation])
        d_tm1 = np.array(timestep.discount)
        losses, gradients = self._v_planning_loss_grad(self._v_network,
                                                    self._o_network,
                                                    self._r_network,
                                                    o_tm1, d_tm1)
        for prev_o_tmn in range(np.prod(self._input_dim)):
            self._v_network[prev_o_tmn] = self._v_planning_opt_update(gradients[prev_o_tmn][0],
                                                             self._v_network[prev_o_tmn])

        losses_and_grads = {"losses": {"loss_v_planning": np.array(np.sum(losses))},
                            }
        self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
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

