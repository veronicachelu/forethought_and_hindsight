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


class OnPolicyDynaAgent(DynaAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(OnPolicyDynaAgent, self).__init__(**kwargs)

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return
        if self.total_steps % self._planning_period == 0:
            for k in range(self._planning_iter):
                transitions = self._replay.sample(self._batch_size)
                # o_tm1, a_tm1, r_t_target, d_t_target, o_t_target
                o_tm1, a_tm1 = transitions
                model_a_tm1 = int(np.argmax(self._q_forward(self._q_parameters, o_tm1), axis=-1))
                transitions[-1] = model_a_tm1

                model_tm1 = self._model_forward(self._model_parameters, o_tm1)
                model_o_t = np.array(jax.vmap(lambda model, a: model[a][:-3])(model_tm1, model_a_tm1))
                model_r_t = np.array(jax.vmap(lambda model, a: model[a][-3])(model_tm1, model_a_tm1))
                model_d_t = np.array(jax.vmap(lambda model, a: jnp.argmax(model[a][-2:], axis=-1))(model_tm1, model_a_tm1),
                                     dtype=np.int32)
                transitions.extend([model_r_t, model_d_t, model_o_t])
                # plan on batch of transitions
                loss, gradient = self._q_loss_grad(self._q_parameters,
                                                   transitions)
                self._q_opt_state = self._q_opt_update(self.total_steps, gradient,
                                                       self._q_opt_state)
                self._q_parameters = self._q_get_params(self._q_opt_state)

                losses_and_grads = {"losses": {"loss_q_planning": np.array(loss),
                                               },
                                    "gradients": {"grad_norm_q_plannin": np.sum(
                                        np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
                self._log_summaries(losses_and_grads, "value_planning")
