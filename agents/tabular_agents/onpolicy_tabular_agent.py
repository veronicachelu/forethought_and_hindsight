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


class OnPolicyTabularAgent(DynaTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(OnPolicyTabularAgent, self).__init__(**kwargs)

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return

        for k in range(self._planning_iter):
            transitions = self._replay.sample(self._batch_size)
            # plan on batch of transitions
            o_tm1, a_tm1 = transitions
            model_a_tm1 = [self._nrng.choice(np.flatnonzero(q_values == np.max(q_values)))
                           for q_values in self._q_network[o_tm1]]
            transitions[-1] = np.array(model_a_tm1)

            loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                        self._model_network,
                                                        transitions)
            self._q_network[o_tm1, model_a_tm1] = self._q_opt_update(gradient,
                                                                     self._q_network[o_tm1, model_a_tm1])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                }
            self._log_summaries(losses_and_grads, "value_planning")

