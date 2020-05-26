import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.agent import Agent
from utils.replay import Replay
from .qt_vanilla import VanillaQT

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TrueFwQT(VanillaQT):
    def __init__(
            self,
            **kwargs
    ):
        super(TrueFwQT, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        self._true_discount = None
        self._create_summary_dirs()

        def q_planning_loss(q_params, fw_o_params, r_params, x, a):
            next_x_logits = fw_o_params[x, a]
            # r = r_params[x]
            target = 0
            next_x_prob = next_x_logits
            q = q_params[x, a]

            for next_x in range(np.prod(self._input_dim)):
                target += next_x_prob[next_x] * \
                (r_params[next_x] + self._true_discount[next_x] * (self._discount ** self._n) * \
                 np.max(q_params[next_x], axis=-1))
            td_error = (target - q)
            loss = td_error ** 2

            return loss, td_error

        self._o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        self._r_network = self._network["model"]["net"][2]

        self._q_planning_loss_grad = q_planning_loss

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
        if self._n == 0:
            return
        if timestep.last():
            return
        # o_t = np.array(timestep.observation)
        x = timestep.observation
        # d_t = np.array(timestep.discount)
        losses = 0
        for a in range(self._nA):
            loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                        self._fw_o_network,
                                                        self._r_network,
                                                        x,
                                                        a)
            losses += loss
            self._q_network[x, a] = self._q_planning_opt_update(
                gradient,
                self._q_network[x, a])

        losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                            "gradients": {}}
        self._log_summaries(losses_and_grads, "q_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
       pass

