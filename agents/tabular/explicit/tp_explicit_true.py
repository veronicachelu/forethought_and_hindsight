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


class TpExplicitTrue(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpExplicitTrue, self).__init__(**kwargs)

        def v_planning_loss(v_params, o, o_tm1, r_tmn1, d_tm1):
            v_tmn = v_params[o_tm1]
            td_error = (r_tmn1 + d_tm1 * self._discount *
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

        if prev_timestep is not None and timestep.reward is not None:
            o_tm2 = np.array([prev_timestep.observation])
            r_tm2 = np.array([timestep.reward])
            d_tm2 = np.array([timestep.discount])
            o_tm1 = np.array([timestep.observation])
            losses, gradients = self._v_planning_loss_grad(self._v_network,
                                                    o_tm1, o_tm2, r_tm2, d_tm2)

            self._v_network[o_tm2] = self._v_planning_opt_update(gradients,
                                                             self._v_network[o_tm2])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(np.mean(losses))},
                                }
            self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return False

    def model_free_train(self):
        return True

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
      pass

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

