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


class TrueBwQT(VanillaQT):
    def __init__(
            self,
            **kwargs
    ):
        super(TrueBwQT, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        self._create_summary_dirs()

        def q_planning_loss(q_params, r_params, x, prev_a, prev_x, d):
            q_tm1 = q_params[prev_x, prev_a]
            r_t = r_params[prev_x, x]
            q_t = q_params[x]
            q_target = r_t + d * (self._discount ** self._n) * np.max(q_t, axis=-1)
            td_error = (q_target - q_tm1)
            return np.mean(td_error ** 2), td_error

        self._o_network = self._network["model"]["net"][0]
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
        if timestep.discount is None:
            return
        o_t = np.array(timestep.observation)
        d_t = np.array(timestep.discount)
        losses = 0
        o_tmn = self._o_network[o_t]
        for prev_o_tmn in range(np.prod(self._input_dim)):
            for prev_a in range(self._nA):
                loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                            self._r_network,
                                                            o_t, prev_o_tmn,
                                                            prev_a, d_t)
                losses += loss
                self._q_network[prev_o_tmn, prev_a] = self._q_planning_opt_update(
                    o_tmn[prev_o_tmn, prev_a] * gradient,
                    self._v_network[prev_o_tmn, prev_a])

        losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                            "gradients": {"grad_norm_q_planning": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
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

