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


class BwQT(VanillaQT):
    def __init__(
            self,
            **kwargs
    ):
        super(BwQT, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1

        def model_loss(o_params,
                       r_params,
                       transitions):
            o_tmn = transitions[0][0]
            a_tmn = transitions[0][1]
            o_t = transitions[-1][-1]

            model_o_tmn = o_params[o_t, a_tmn]
            o_target = np.eye(np.prod(self._input_dim))[o_tmn]
            o_loss = self._ce(self._log_softmax(model_o_tmn), o_target)
            o_tmn_probs = self._softmax(model_o_tmn)
            o_tmn_probs[o_tmn] -= 1
            o_error = - o_tmn_probs
            model_r_t = r_params[o_tmn][o_t]
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_error = (r_t_target - model_r_t)
            r_loss = np.mean(r_error ** 2)

            total_error = o_loss + r_loss
            return (total_error, o_loss, r_loss), (o_error, r_error)

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

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params: \
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if self._n == 0:
            return
        if len(self._sequence) >= self._n:
            o_tmn = self._sequence[0][0]
            a_tmn = self._sequence[0][1]
            o_t = self._sequence[-1][-1]
            losses, gradients = self._model_loss_grad(self._o_network,
                                                      self._r_network,
                                                      self._sequence)
            self._o_network[o_t, a_tmn], self._r_network[o_tmn, o_t] = \
                self._model_opt_update(gradients, [self._o_network[o_t, a_tmn],
                                                   self._r_network[o_tmn, o_t]])
            total_loss, o_loss, r_loss = losses
            o_grad, r_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": o_loss,
                "r_loss": r_loss,
                "o_grad": o_grad,
                "r_grad": r_grad,
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False


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
        for prev_o_tmn in range(np.prod(self._input_dim)):
            for prev_a in range(self._nA):
                o_tmn = self._softmax(self._o_network[o_t, prev_a])
                loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                            self._r_network,
                                                            o_t, prev_a, prev_o_tmn,
                                                            d_t)
                losses += loss
                self._q_network[prev_o_tmn, prev_a] = self._q_planning_opt_update(
                    o_tmn[prev_o_tmn] * gradient,
                    self._q_network[prev_o_tmn, prev_a])

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
        self._sequence.append([timestep.observation,
                               action,
                               new_timestep.reward,
                               new_timestep.discount,
                               new_timestep.observation])
        if new_timestep.discount == 0:
            self._should_reset_sequence = True

