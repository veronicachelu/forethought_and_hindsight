import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp
from copy import deepcopy
from agents.agent import Agent
from utils.replay import Replay
from .qt_vanilla import VanillaQT
import itertools
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

        self._run_mode = "{}".format(self._run_mode)

        self._create_summary_dirs()

        def model_loss(o_params,
                       r_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            a_tmn_target = transitions[0][1]
            o_t = transitions[-1][-1]

            oa_target_index = np.ravel_multi_index([o_tmn_target, a_tmn_target], (np.prod(self._input_dim), 4))
            model_o_tmn = o_params[o_t]
            oa_target = np.eye(np.prod(self._input_dim)*4)[oa_target_index]
            o_loss = self._ce(self._log_softmax(model_o_tmn), oa_target)
            o_tmn_probs = self._softmax(model_o_tmn)
            o_tmn_probs[oa_target_index] -= 1
            o_error = - o_tmn_probs
            model_r_t = r_params[o_t]

            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_error = (r_t_target - model_r_t)
            r_loss = np.mean(r_error ** 2)

            total_error = o_loss + r_loss
            return (total_error, o_loss, r_loss), (o_error, r_error)

        def q_planning_loss(q_params, r_params, x, prev_a, prev_x, r_t, d):
            q_tm1 = q_params[prev_x, prev_a]
            r_t = r_params[x]
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
            # a_tmn = self._sequence[0][1]
            o_t = self._sequence[-1][-1]

            losses, gradients = self._model_loss_grad(self._o_network,
                                                      self._r_network,
                                                      self._sequence)
            self._o_network[o_t], self._r_network[o_t] = \
                self._model_opt_update(gradients, [self._o_network[o_t],
                                                   self._r_network[o_t]])
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
        o_t = timestep.observation
        d_t = timestep.discount
        r_t = timestep.reward
        losses = 0

        # if self._q_network[15, 0] != 0:
        #     print("BEFORE####### q_t(15,0) == {} ########".format(self._q_network[15, 0]))
        # clone = deepcopy(self._q_network)
        probs = self._softmax(self._o_network[o_t])
        # to_update = probs.argsort()[-self._top_n:][::-1]
        to_update = probs.argsort()[::-1]
        for oa_index in to_update:
            prev_o, prev_a = np.unravel_index(oa_index, (np.prod(self._input_dim), 4))
            # if prev_o == o_t:
            #     continue
            # for prev_o, prev_a in itertools.product(range(np.prod(self._input_dim)), range(self._nA)):
            #     oa_index = np.ravel_multi_index([prev_o, prev_a], (48, 4))
            loss, gradient = self._q_planning_loss_grad(self._q_network,
                                                        self._r_network,
                                                        o_t, prev_a, prev_o,
                                                        r_t,
                                                        d_t)
            losses += loss
            self._q_network[prev_o, prev_a] = self._q_planning_opt_update(
                probs[oa_index] * gradient,
                self._q_network[prev_o, prev_a])

            # if gradient > 0:
            # if (prev_o, prev_a) == (15, 0) and clone[prev_o, prev_a] > 0:
            #     print("Updating q({},{}) from {} with p {} g {} \n "
            #           "(prev q_t({},{})={}, next q_t({},{})={}"
            #           " target q_t({})={} r={})".format(
            #         prev_o, prev_a, o_t, probs[oa_index], gradient,
            #         prev_o, prev_a, clone[prev_o, prev_a],
            #         prev_o, prev_a, self._q_network[prev_o, prev_a],
            #         o_t, np.max(self._q_network[o_t]), r_t,
            #     ))

        # if self._q_network[15, 0] != 0:
        #     print("AFTER####### q_t(15,0) == {} ########".format(self._q_network[15, 0]))

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

    def update_hyper_params(self, episode, total_episodes):
        self._lr = self._initial_lr * ((total_episodes - episode) / total_episodes)
        self._lr_model = self._initial_lr_model * ((total_episodes - episode) / total_episodes)

        steps_left = total_episodes + 0 - episode
        bonus = (self._initial_epsilon - self._final_epsilon) * steps_left / total_episodes
        bonus = np.clip(bonus, 0., self._initial_epsilon - self._final_epsilon)
        self._epsilon = self._final_epsilon + bonus
        if self._logs is not None:
            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                tf.summary.scalar("train/epsilon",
                                  self._epsilon, step=ep)
                self.writer.flush()

    def get_model_for_all_states(self, all_states):
        state_action = np.transpose(
            np.reshape(self._softmax(self._o_network[all_states]), (-1, np.prod(self._input_dim), 4)),
            axes=[0, 2, 1])
        return state_action


