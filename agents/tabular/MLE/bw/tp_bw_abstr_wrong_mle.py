import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from agents.tabular.MLE.tp_vanilla import TpVanilla
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class TpBwAbstrWMLE(TpVanilla):
    def __init__(
            self,
            **kwargs
    ):

        super(TpBwAbstrWMLE, self).__init__(**kwargs)
        self._sequence = []
        self._should_reset_sequence = False

        self._o_network = self._network["model"]["net"][0]
        self._fw_o_network = self._network["model"]["net"][1]
        self._r_network = self._network["model"]["net"][2]

        self._abstraction = [
                [0.5, 0.5, 0, 0, 0, 0],
                [0, 0, 0.5, 0.5, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ]
        def model_loss(v_params, o_params, r_params, transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]
            abstract_o_tmn = self._abstract(o_tmn_target)
            abstract_o_t = self._abstract(o_t)
            abstr_shape = 4
            model_o_tmn = o_params[abstract_o_t]
            P = self._softmax(model_o_tmn)
            o_target = np.eye(abstr_shape)[abstract_o_tmn]

            abstract_o_loss = self._ce(self._log_softmax(model_o_tmn), o_target)

            abstract_o_tmn_probs = P
            abstract_o_tmn_probs[abstract_o_tmn] -= 1
            # o_tmn_probs /= len(o_tmn_probs)
            abstract_o_error = - abstract_o_tmn_probs

            # o_target = np.eye(np.prod(self._input_dim))[o_tmn_target] - o_tmn
            # o_error = o_target - o_tmn
            # o_loss = np.mean(o_error ** 2)

            r_tmn = r_params[o_tmn_target, o_t]

            r_tmn_target = 0
            for i, t in enumerate(transitions):
                r_tmn_target += (self._discount ** i) * t[2]

            r_error = r_tmn_target - r_tmn
            r_loss = np.mean(r_error ** 2)

            total_error = abstract_o_loss + r_loss

            ### PAML LOSS ###
            x_shape = np.prod(self._input_dim)
            td_errors = np.array(r_params[np.arange(x_shape), o_t] +
                                 (self._discount ** self._n) * \
                                 v_params[o_t] - v_params[np.arange(x_shape)])

            abstract_td_errors = np.matmul(self._abstraction, td_errors)
            abstract_o_tmn = self._abstract(o_tmn_target)
            abstract_v_params = np.matmul(self._abstraction, v_params)

            model_Delta = P * abstract_td_errors

            real_abstract_td_error = r_tmn_target + (self._discount ** self._n) * \
                                                    abstract_v_params[abstract_o_t] - abstract_v_params[abstract_o_tmn]

            real_Delta = np.eye(abstr_shape)[abstract_o_t] * real_abstract_td_error

            # cov = np.outer(td_errors, P) - np.outer(P * td_errors, P)
            cov = P * np.diag(abstract_td_errors) - np.outer(P * abstract_td_errors, P)
            PAML_loss = np.mean((real_Delta - model_Delta) ** 2)

            return (total_error, abstract_o_loss, r_loss, PAML_loss), (abstract_o_error, r_error)

        self._model_loss_grad = model_loss
        self._model_opt_update = lambda gradients, params:\
            [param + self._lr_model * grad for grad, param in zip(gradients, params)]

        def v_planning_loss(v_params, r_params, o_t, o_tmn, d_t):
            abstract_o_t = self._abstract(o_t)
            abstract_o_tmn = self._abstract(o_tmn)
            abstract_v_params = np.matmul(self._abstraction, v_params)
            # v_tmn = v_params[o_tmn]
            v_tmn = abstract_v_params[abstract_o_tmn]
            v_t = abstract_v_params[abstract_o_t]
            r_tmn = r_params[o_tmn, o_t]
            td_error = (r_tmn + d_t * (self._discount ** self._n) *
                        v_t - v_tmn)

            loss = td_error ** 2

            return loss, td_error

        self._v_planning_loss_grad = v_planning_loss

        def v_loss(v_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            abstract_o_t = self._abstract(o_t)
            abstract_o_tmn = self._abstract(o_tm1)
            abstract_v_params = np.matmul(self._abstraction, v_params)

            v_tm1 = abstract_v_params[abstract_o_tmn]
            v_t = abstract_v_params[abstract_o_t]
            v_target = r_t + d_t * self._discount * v_t
            td_error = (v_target - v_tm1)
            return np.mean(td_error ** 2), td_error

        self._v_loss_grad = v_loss

    def _abstract(self, x):
        # return x // 2
        mapping = [0, 0, 1, 1, 2, 3]
        return mapping[x]

    def _aggregate(self, x):
        # return x * self._M + np.arange(self._M)

        mapping = [[0, 1], [2, 3], [0, 1], [2, 3], [4], [5]]
        return mapping[x]

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
            o_t = self._sequence[-1][-1]
            losses, gradients = self._model_loss_grad(self._v_network,
                                                      self._o_network,
                                                      self._r_network,
                                                      self._sequence)

            abstract_o_t = self._abstract(o_t)
            self._o_network[abstract_o_t], self._r_network[o_tmn, o_t] = \
                self._model_opt_update(gradients, [self._o_network[abstract_o_t],
                                                   self._r_network[o_tmn, o_t]])
            total_loss, o_loss, r_loss, PAML_loss = losses
            o_grad, r_grad = gradients
            o_grad = np.linalg.norm(np.asarray(o_grad), ord=2)
            o_norm = np.max(np.linalg.norm(np.asarray(self._o_network), ord=2, axis=-1))
            losses_and_grads = {"losses": {
                "loss": total_loss,
                "o_loss": PAML_loss,
                "r_loss": r_loss,
                "o_grad": o_grad,
                "o_norm": o_norm,
                "r_grad": r_grad,
            },
            }
            # print("max_param_norm {}".format(o_norm))
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
        if timestep.discount is None:
            return

        o_t = np.array(timestep.observation)
        d_t = np.array(timestep.discount)
        abstract_o_t = self._abstract(o_t)
        P = self._softmax(self._o_network[abstract_o_t])

        delta = np.zeros(shape=(np.prod(self._input_dim),))
        losses = np.zeros(shape=(np.prod(self._input_dim),))
        for prev_o_tmn in range(np.prod(self._input_dim)):
            loss, gradient = self._v_planning_loss_grad(self._v_network,
                                                           self._r_network,
                                                           o_t, prev_o_tmn, d_t)
            delta[prev_o_tmn] = gradient
            losses[prev_o_tmn] = loss
        abstract_v = np.matmul(self._abstraction, delta)
        abstract_loss = np.matmul(self._abstraction, losses)
        for prev_o_tmn in range(4):
            all_in_abstraction = self._aggregate(prev_o_tmn)
            self._v_network[all_in_abstraction] = self._v_planning_opt_update(
                P[prev_o_tmn] * abstract_v[prev_o_tmn],
                self._v_network[all_in_abstraction])

        losses_and_grads = {"losses": {"loss_v_planning": np.sum(abstract_loss)},
                            }
        self._log_summaries(losses_and_grads, "value_planning")

    def model_based_train(self):
        return True

    def model_free_train(self):
        # return False
        return True

    def load_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            if os.path.exists(checkpoint):
                to_load = np.load(checkpoint, allow_pickle=True)[()]
                self.episode = to_load["episode"]
                self.total_steps = to_load["total_steps"]
                self._v_network = to_load["v_parameters"]
                self._o_network = to_load["o_parameters"]
                self._r_network = to_load["r_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")


    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        o_tm1 = np.array(timestep.observation)
        a_tm1 = np.array(action)
        r_t = np.array(new_timestep.reward)
        d_t = np.array(new_timestep.discount)
        o_t = np.array(new_timestep.observation)
        transitions = [o_tm1, a_tm1, r_t, d_t, o_t]

        loss, gradient = self._v_loss_grad(self._v_network, transitions)
        abstract_o_t = self._abstract(o_tm1)
        all_in_abstraction = self._aggregate(abstract_o_t)
        self._v_network[all_in_abstraction] = self._v_opt_update(gradient, self._v_network[all_in_abstraction])

        losses_and_grads = {"losses": {"loss_v": np.array(loss)},
                            }
        self._log_summaries(losses_and_grads, "value")

    def save_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "v_parameters": self._v_network,
                "o_parameters": self._o_network,
                "r_parameters": self._r_network,
            }
            np.save(checkpoint, to_save)
            print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                               self.total_steps,
                                                                               checkpoint))

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

    def _log_summaries(self, losses_and_grads, summary_name):
        if self._logs is not None:
            losses = losses_and_grads["losses"]

            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), losses[k], step=self.total_steps)
                self.writer.flush()

    def update_hyper_params(self, episode, total_episodes):
        self._lr = self._initial_lr * ((total_episodes - episode) / total_episodes)
        self._lr_model = self._initial_lr_model * ((total_episodes - episode) / total_episodes)
        self._lr_planning = self._initial_lr_planning * ((total_episodes - episode) / total_episodes)