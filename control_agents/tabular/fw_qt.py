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


class FwQT(VanillaQT):
    def __init__(
            self,
            **kwargs
    ):
        super(FwQT, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False
        self._n = 1
        self._true_discount = None
        self._create_summary_dirs()

        def model_loss(fw_o_params,
                       r_params,
                       d_params,
                       transitions):
            o_tmn = transitions[0][0]
            a_tmn = transitions[0][1]
            d_t = transitions[0][3]
            o_t_target = transitions[-1][-1]

            # forward
            model_o_t = fw_o_params[o_tmn, a_tmn]
            o_target = np.eye(np.prod(self._input_dim))[o_t_target]
            fw_o_loss = self._ce(self._log_softmax(model_o_t), o_target)
            o_t_probs = self._softmax(model_o_t)
            o_t_probs[o_t_target] -= 1
            fw_o_error = - o_t_probs
            model_r_t = r_params[o_t_target]

            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]

            r_error = (r_tmn_target - model_r_t)
            # print("grad is {}".format(r_error))
            r_loss = np.mean(r_error ** 2)

            model_d_t = d_params[o_t_target]
            d_target = np.eye(2)[int(d_t)]
            d_loss = self._ce(self._log_softmax(model_d_t), d_target)
            d_t_probs = self._softmax(model_d_t)
            d_t_probs[int(d_t)] -= 1
            d_error = - d_t_probs

            total_error = fw_o_loss + r_loss + d_loss
            return (total_error, fw_o_loss, r_loss, d_loss), (fw_o_error, r_error, d_error)


        def q_planning_loss(q_params, fw_o_params, r_params, d_params, x, a):
            next_x_logits = fw_o_params[x, a]
            # next_x_logits = self._true_fw_o_network[x, a]
            # r = self._true_r_network[x]
            # r = r_params[x]
            target = 0
            next_x_prob = self._softmax(next_x_logits)
            # next_x_prob = next_x_logits
            q = q_params[x, a]

            for next_x in range(np.prod(self._input_dim)):
                d_values = d_params[next_x]
                d = np.argmax(d_values)
                target += next_x_prob[next_x] * \
                (r_params[next_x] + d * (self._discount ** self._n) * \
                 np.max(q_params[next_x], axis=-1))
            td_error = (target - q)
            loss = td_error ** 2

            return loss, td_error

        # self._o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        # self._r_network = self._network["model"]["net"][2]
        self._r_network = self._network["model"]["net"][3]
        self._d_network = self._network["model"]["net"][4]

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params: \
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        self._q_planning_loss_grad = q_planning_loss

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
            if self._sequence[0][2] == 1:
                print("reward 1")
            losses, gradients = self._model_loss_grad(self._fw_o_network,
                                                      self._r_network,
                                                      self._d_network,
                                                      self._sequence)
            self._fw_o_network[o_tmn, a_tmn],\
            self._r_network[o_t],\
            self._d_network[o_t] = \
                self._model_opt_update(gradients, [self._fw_o_network[o_tmn, a_tmn],
                                               self._r_network[o_t],
                                               self._d_network[o_t]])
            total_loss, o_loss, r_loss, d_loss = losses
            o_grad, r_grad, d_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            d_grad = np.linalg.norm(np.asarray(d_grad), ord=2)
            if r_grad > 0:
                print(r_grad)
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": o_loss,
                "r_loss": r_loss,
                "d_loss": d_loss,
                "o_grad": o_grad,
                "r_grad": r_grad,
                "d_grad": d_grad,
            },
            }
            self._log_summaries(losses_and_grads, "model")
            self._sequence = self._sequence[1:]

        if self._should_reset_sequence:
            self._sequence = []
            self._should_reset_sequence = False

    def get_model_for_all_states(self, all_states):
        logits = self._fw_o_network[all_states]
        probs = self._softmax(logits, axis=1)
        state_action = np.reshape(probs, (-1, 4, np.prod(self._input_dim)))
        return state_action

    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        if self._n == 0:
            return
        if timestep.last():
            return
        x = timestep.observation
        # d_t = np.array(timestep.discount)
        losses = 0
        for a in range(self._nA):
            loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                        self._fw_o_network,
                                                        self._r_network,
                                                        self._d_network,
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
        self._sequence.append([timestep.observation,
                               action,
                               new_timestep.reward,
                               new_timestep.discount,
                               new_timestep.observation])

        if new_timestep.discount == 0:
            self._should_reset_sequence = True

