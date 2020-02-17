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


class PriorityDynaTabularAgent(DynaTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(PriorityDynaTabularAgent, self).__init__(**kwargs)

        def td_error(q_params, model_params, transitions):
            o_tm1, a_tm1 = transitions
            o_t = model_params[o_tm1, a_tm1, :-3]
            r_t = model_params[o_tm1, a_tm1, -3]
            d_t = np.argmax(model_params[o_tm1, a_tm1, -2:], axis=-1)

            q_tm1 = q_params[o_tm1, a_tm1]

            q_target = r_t
            divisior = np.sum(o_t, axis=-1, keepdims=True)
            o_t = np.divide(o_t, divisior, out=np.zeros_like(o_t), where=np.all(divisior != 0))
            for next_o_t in range(np.prod(self._input_dim)):
                q_t = q_params[next_o_t]
                q_target += d_t * self._discount * o_t[:, next_o_t] * np.max(q_t, axis=-1)
            td_error = q_target - q_tm1

            return td_error

        self._td_error = td_error

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return

        for k in range(self._planning_iter):
            priority_transitions = self._replay.peek_n_priority(self._batch_size)
            priority = priority_transitions[0]
            transitions = priority_transitions[1:]
            # plan on batch of transitions
            o_tm1, a_tm1 = transitions

            loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                        self._model_network,
                                                        transitions)
            self._q_network[o_tm1, a_tm1] = self._q_opt_update(gradient, self._q_network[o_tm1, a_tm1])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                }
            self._log_summaries(losses_and_grads, "value_planning")

            td_error = np.asarray(self._td_error(self._q_network,
                                                 self._model_network,
                                                 transitions))
            priority = np.abs(td_error)
            o_tm1, a_tm1 = transitions
            for i in range(len(o_tm1)):
                self._replay.add([
                    priority[i],
                    o_tm1[i],
                    a_tm1[i],
                ])

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        transitions = [np.array([timestep.observation]),
                       np.array([action])]
        td_error = np.asarray(self._td_error(self._q_network,
                                             self._model_network,
                                             transitions))
        priority = np.abs(td_error)
        # Add this states and actions to replay.
        self._replay.add([
            priority,
            timestep.observation,
            action
        ])

