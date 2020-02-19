from typing import Any, Callable, Sequence
import os
from utils.replay import Replay

import dm_env
from dm_env import specs

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import numpy as np
from agents.dyna_agent import DynaAgent
import tensorflow as tf

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class Planning2ExploreAgent(DynaAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(Planning2ExploreAgent, self).__init__(**kwargs)

    def planning_update(
            self
    ):
        pass

    def _structured_exploration(self,
                               observation: int,
                               depth: int):
        q_ts = self._q_network(self._q_parameters, observation[None, :])[0]
        if depth > 0:
            depth -= 1
            o_tm1 = observation

            q_ts = []
            for a_tm1 in range(self._nA):
                model_tm1 = self._model_network(self._model_parameters, o_tm1[None, :])[0]
                o_t, r_t, d_t = model_tm1[a_tm1][:-3],  model_tm1[a_tm1][-3], jnp.argmax(model_tm1[a_tm1][-2:], axis=-1)
                backup_q, backup_q = self._structured_exploration(o_t, depth)
                q_t = r_t + d_t * self._discount * backup_q
                q_ts.append(q_t)

            q_ts = np.array(q_ts)

        max_a_tm1 = jnp.argmax(q_ts)
        max_q_t = jnp.max(q_ts)

        return max_a_tm1, max_q_t


    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        # Epsilon-greedy policy if not test policy.
        if not eval and self._nrng.rand() < self._epsilon:
            return self._nrng.randint(self._nA)
        a, _ = self._structured_exploration(timestep.observation, self._planning_depth)
        return a