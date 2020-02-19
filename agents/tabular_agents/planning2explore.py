import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular_agents.dyna_tabular_agent import DynaTabularAgent
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class Planning2ExploreTabularAgent(DynaTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(Planning2ExploreTabularAgent, self).__init__(**kwargs)

    def planning_update(
            self
    ):
        pass

    def _structured_exploration(self,
                               observation: int,
                               depth: int):

        if depth == 0:
            q_ts = self._q_network[observation]
        else:
            depth -= 1
            o_tm1 = observation

            q_ts = []
            for a_tm1 in range(self._nA):
                prob_o_t = self._model_network[o_tm1, a_tm1, :-3]
                r_t = self._model_network[o_tm1, a_tm1, -3]
                d_t = np.argmax(self._model_network[o_tm1, a_tm1, -2:], axis=-1)

                q_t = r_t
                divisior = np.sum(prob_o_t, axis=-1, keepdims=True)
                prob_o_t = np.divide(prob_o_t, divisior, out=np.zeros_like(prob_o_t), where=np.all(divisior != 0))
                for o_t in range(np.prod(self._input_dim)):
                    backup_q, backup_q = self._structured_exploration(o_t, depth)
                    q_t += d_t * self._discount * prob_o_t[o_t] * backup_q

                q_ts.append(q_t)

            q_ts = np.array(q_ts)

        max_a_tm1 = self._nrng.choice(np.flatnonzero(q_ts == np.max(q_ts)))
        max_q_t = self._q_network[observation, max_a_tm1]

        return max_a_tm1, max_q_t

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        # Epsilon-greedy policy if not test policy.
        if not eval and self._nrng.rand() < self._epsilon:
            return self._nrng.randint(self._nA)
        # q_values = self._q_network[timestep.observation]
        # a = self._nrng.choice(np.flatnonzero(q_values == np.max(q_values)))
        a, _ = self._structured_exploration(timestep.observation, self._planning_depth)
        return a


