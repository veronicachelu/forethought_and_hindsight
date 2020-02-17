import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular_agents.priority_dyna_tabular_agent import PriorityDynaTabularAgent
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class PredecessorsDynaTabularAgent(PriorityDynaTabularAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(PredecessorsDynaTabularAgent, self).__init__(**kwargs)

        def reverse_model_loss(model_params, transitions):
            o_tm1_target, a_tm1_target, r_t, d_t, o_t = transitions
            model_o_t = model_params[0][o_t, a_tm1_target]
            model_a_t_logits = model_params[1][o_t, o_tm1_target]
            model_a_t_probs = self._softmax(model_a_t_logits)
            o_error = np.eye(np.prod(self._input_dim))[o_tm1_target] - model_o_t
            a_tm1_target = np.array(a_tm1_target, dtype=np.int32)
            nll = np.take_along_axis(model_a_t_logits, np.expand_dims(a_tm1_target, axis=-1), axis=1)
            a_t_error = - np.mean(nll)
            model_a_t_probs[np.arange(len(model_a_t_probs)), a_tm1_target] -= 1
            model_a_t_probs /= len(model_a_t_probs)
            model_a_t_probs = -model_a_t_probs
            o_loss = np.mean(o_error ** 2)
            total_error = o_loss + a_t_error
            return (total_error, o_loss, a_t_error), (o_error, model_a_t_probs)

        # This function computes dL/dTheta
        self._reverse_model_loss_grad = reverse_model_loss
        self._reverse_model_opt_update = lambda gradients, params: \
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def q_planning_loss(q_params, model_params, reverse_model_params, transitions):
            o_tm1, a_tm1 = transitions
            o_t = q_params[o_tm1, a_tm1, :-3]
            r_t = q_params[o_tm1, a_tm1, -3]
            d_t = np.argmax(model_params[o_tm1, a_tm1, -2:], axis=-1)

            q_tm1 = self._q_network[o_tm1, a_tm1]

            q_target = r_t
            divisior = np.sum(o_t, axis=-1, keepdims=True)
            o_t = np.divide(o_t, divisior, out=np.zeros_like(o_t), where=np.all(divisior != 0))
            for next_o_t in range(np.prod(self._input_dim)):
                q_t = q_params[next_o_t]
                q_target += d_t * self._discount * o_t[:, next_o_t] * np.max(q_t, axis=-1)
            td_error = q_target - q_tm1

            model_o_a_tm2 = np.minimum(reverse_model_params[o_tm1], 0)

            for i in range(len(model_o_a_tm2)):
                divisor = np.sum(model_o_a_tm2[i], keepdims=True)
                if divisor == 0:
                    continue

                model_o_a_tm2_prob = np.divide(model_o_a_tm2[i], divisor,
                                          out=np.zeros_like(model_o_a_tm2[i]),
                                          where=np.all(divisor != 0))
                model_o_tm2, model_a_tm2 = np.unravel_index(np.argmax(np.random.multinomial(1, model_o_a_tm2_prob.ravel())),
                                             [np.prod(self._input_dim), self._nA])

                model_r_tm1 = q_params[model_o_tm2, model_a_tm2, -3]
                model_d_tm1 = np.argmax(model_params[model_o_tm2, model_a_tm2, -2:], axis=-1)
                q_tm1 = q_params[o_tm1]
                target = model_r_tm1 + model_d_tm1 * self._discount * np.max(q_tm1, axis=-1)
                td_error = target - q_params[model_o_tm2, model_a_tm2]

                # o_tm1, a_tm1 = transitions
                # for i in range(len(o_tm1)):
                #     self._replay.add([
                #         priority[i],
                #         o_tm1[i],
                #         a_tm1[i],
                #     ])
            return np.mean(td_error ** 2), td_error

        self._q_planning_loss_grad = q_planning_loss


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
                                                        self._reverse_model_network,
                                                        transitions)
            self._q_network[o_tm1, a_tm1] = self._q_opt_update(gradient, self._q_network[o_tm1, a_tm1])

            losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
                                }
            self._log_summaries(losses_and_grads, "value_planning")

            td_error = np.asarray(self._td_error(transitions))
            priority = np.abs(td_error)
            o_tm1, a_tm1 = transitions
            for i in range(len(o_tm1)):
                self._replay.add([
                    priority[i],
                    o_tm1[i],
                    a_tm1[i],
                ])

    # python
    # main.py - -run_mode = dyna - -model_class = linear -
    # -obs_type = onehot - -lr = 1e-3 - -lr_model = 1e-3
    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        super().model_update(timestep, action, new_timestep)
        o_tm1 = np.array([timestep.observation])
        a_tm1 = np.array([action])
        r_t = np.array([new_timestep.reward])
        d_t = np.array([new_timestep.discount])
        o_t = np.array([new_timestep.observation])
        transitions = [o_tm1, a_tm1, r_t, d_t, o_t]

        losses, gradients = self._reverse_model_loss_grad(self._reverse_model_network,
                                                          transitions)
        self._reverse_model_network[0][o_t, a_tm1] = self._model_opt_update(gradients[0],
                                                                        self._reverse_model_network[0][o_t, a_tm1])
        self._reverse_model_network[1][o_t, o_tm1] = self._model_opt_update(gradients[1],
                                                                            self._reverse_model_network[1][o_t, o_tm1])
        total_loss_o, total_loss_a = losses
        grad_o, grad_a = gradients
        o_grad = np.sum(np.linalg.norm(np.asarray(grad_o), ord=2), axis=-1)
        a_grad = np.sum(np.linalg.norm(np.asarray(grad_a), ord=2), axis=-1)
        losses_and_grads = {"losses": {
            "loss_reverse_o": total_loss_o,
            "loss_reverse_a": total_loss_a,
            "reverse_o_grad": o_grad,
            "reverse_a_grad": a_grad,
        },
        }
        self._log_summaries(losses_and_grads, "reverse_model")

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

