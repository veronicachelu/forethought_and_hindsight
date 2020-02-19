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


class PriorityDynaAgent(DynaAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(DynaAgent, self).__init__(**kwargs)

        self._replay._alpha = 1.0
        self._replay._initial_beta = 1.0
        self._replay._beta = self._replay._initial_beta

        def priority(self,
                     transitions):
            o_tm1, a_tm1 = transitions
            model_tm1 = self._model_forward(self._model_parameters, o_tm1)
            model_o_t, model_r_t, model_d_t = jax.vmap(lambda model, a:
                                                       (model[a][:-2],
                                                        model[a][-2],
                                                        random.bernoulli(self._rng,
                                                                         p=jax.nn.sigmoid(model[a][-1])))
                                                       )(model_tm1, a_tm1)
            model_o_t, model_r_t, model_d_t = lax.stop_gradient(model_o_t), \
                                              lax.stop_gradient(model_r_t), \
                                              lax.stop_gradient(model_d_t)
            q_tm1 = self._q_network(self._q_parameters, o_tm1)
            q_t = self._q_network(self._q_parameters, model_o_t)
            q_target = model_r_t + model_d_t * self._discount * jnp.max(q_t, axis=-1)
            q_a_tm1 = jax.vmap(lambda q, a: q[a])(q_tm1, a_tm1)
            td_error = lax.stop_gradient(q_target) - q_a_tm1

            td_error = np.abs(td_error)
            return td_error

        self._priority = jax.jit(priority)

    def update_hyper_params(self, step, total_steps):
        steps_left = total_steps - step
        bonus = (self._replay._initial_beta - 1.0) * steps_left / total_steps
        self._replay._beta = 1.0 - bonus

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return
        if self.total_steps % self._planning_period == 0:
            for k in range(self._planning_iter):
                weights, priority_transitions = self._replay.sample_priority(self._batch_size)
                priority, transitions = priority_transitions
                # plan on batch of transitions
                loss, gradient = self._q_planning_loss_grad(self._q_parameters,
                                                            transitions)
                self._q_opt_state = self._q_opt_update(self.total_steps, gradient * weights,
                                                       self._q_opt_state)
                self._q_parameters = self._q_get_params(self._q_opt_state)

                losses_and_grads = {"losses": {"loss_q_planning": np.array(loss),
                                               },
                                    "gradients": {"grad_norm_q_plannin": np.sum(
                                        np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
                self._log_summaries(losses_and_grads, "value_planning")


    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        transitions = [np.array([timestep.observation]),
                       np.array([action])]
        priority = np.asarray(self._priority(transitions))
        # Add this states and actions to replay.
        self._replay.add([
            priority,
            timestep.observation,
            action
        ])

