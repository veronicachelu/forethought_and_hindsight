import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular_agents.replay_tabular_agent import ReplayTabularAgent
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class PriorityReplayTabularAgent(ReplayTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(PriorityReplayTabularAgent, self).__init__(**kwargs)

        self._replay._alpha = 1.0
        self._replay._initial_beta = 1.0
        self._replay._beta = self._replay._initial_beta

        def priority(q_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = q_params[o_tm1, a_tm1]
            q_t = q_params[o_t]
            q_target = r_t + d_t * self._discount * np.max(q_t, axis=-1)
            td_error = q_target - q_tm1
            td_error = np.abs(td_error)
            # td_error[np.all([q_target == 0, q_tm1 == 0])] = -np.infty
            return td_error

        self._priority = priority

    def update_hyper_params(self, step, total_steps):
        steps_left = total_steps - step
        bonus = (self._replay._initial_beta - 1.0) * steps_left / total_steps
        self._replay._beta = 1.0 - bonus

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return

        for k in range(self._planning_iter):
            weights, priority_transitions = self._replay.sample_priority(self._batch_size)
            # plan on batch of transitions
            priority = priority_transitions[0]
            transitions = priority_transitions[1:]
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            loss, gradient = self._q_loss_grad(self._q_network,
                                               transitions)
            self._q_network[o_tm1, a_tm1] = self._q_opt_update(gradient * weights, self._q_network[o_tm1, a_tm1])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                }
            self._log_summaries(losses_and_grads, "value_planning")


    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        transitions = [np.array([timestep.observation]),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array([new_timestep.observation])]
        priority = np.asarray(self._priority(self._q_network, transitions))
        # Add this transition to replay.
        self._replay.add([
            priority,
            timestep.observation,
            action,
            new_timestep.reward,
            new_timestep.discount,
            new_timestep.observation,
        ])

