from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import numpy as np
from agents.agent import Agent
import tensorflow as tf
import rlax
from basis.feature_mapper import FeatureMapper

from control_agents.vanilla_q import VanillaQ
NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class FwQ(VanillaQ):
    def __init__(
            self,
            **kwargs
    ):
        super(FwQ, self).__init__(**kwargs)

        self._sequence = []
        self._should_reset_sequence = False

        def q_loss(q_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = self._q_network(q_params, o_tm1)
            q_t = self._q_network(q_params, o_t)
            batch_q_learning = jax.vmap(rlax.q_learning)
            td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
            return jnp.mean(td_error ** 2)

        # Internalize the networks.
        self._q_network = network["qvalue"]["net"]
        self._q_parameters = network["qvalue"]["params"]

        # This function computes dL/dTheta
        self._q_loss_grad = jax.jit(jax.value_and_grad(q_loss))
        self._q_forward = jax.jit(self._q_network)

        # Make an Adam optimizer.
        q_opt_init, q_opt_update, q_get_params = optimizers.adam(step_size=self._lr)
        self._q_opt_update = jax.jit(q_opt_update)
        self._q_opt_state = q_opt_init(self._q_parameters)
        self._q_get_params = q_get_params

    def _get_features(self, o):
        if self._feature_mapper is not None:
            return self._feature_mapper.get_features(o, self._nrng)
        else:
            return o

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        if not eval and self._nrng.rand() < self._epsilon:
            return self._nrng.randint(self._nA)
        features = self._get_features(timestep.observation[None, ...])
        q_values = self._q_forward(self._q_parameters, features)[0]
        return int(np.argmax(q_values))

    def v_function(self, o):
        features = self._get_features(o[None, ...])
        q_values = self._q_forward(self._q_parameters, features)
        return np.max(q_values, -1)[0]

    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        features = self._get_features([timestep.observation])
        next_features = self._get_features([new_timestep.observation])
        transitions = [np.array(features),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array(next_features)]

        loss, gradient = self._q_loss_grad(self._q_parameters,
                                transitions)
        self._q_opt_state = self._q_opt_update(self.episode, gradient,
                                               self._q_opt_state)
        self._q_parameters = self._q_get_params(self._q_opt_state)

        losses_and_grads = {"losses": {"loss_q": np.array(loss)},
                            "gradients": {"grad_norm_q": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        self._log_summaries(losses_and_grads, "value")

    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ):
        pass
        # replay_sample = self._replay.sample(32)
        # features = self._get_features([replay_sample[0]])
        # next_features = self._get_features([replay_sample[4]])
        # transitions = [np.array(features),
        #                np.array([replay_sample[1]]),
        #                np.array([replay_sample[2]]),
        #                np.array([replay_sample[3]]),
        #                np.array(next_features)]
        #
        # loss, gradient = self._q_loss_grad(self._q_parameters,
        #                                    transitions)
        # self._q_opt_state = self._q_opt_update(self.episode, gradient,
        #                                        self._q_opt_state)
        # self._q_parameters = self._q_get_params(self._q_opt_state)
        #
        # losses_and_grads = {"losses": {"loss_v_planning": np.array(loss),
        #                                },
        #                     "gradients": {"grad_norm_v_planning": np.sum(
        #                         np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        # self._log_summaries(losses_and_grads, "value_planning")
        #

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def load_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            if os.path.exists(checkpoint):
                to_load = np.load(checkpoint, allow_pickle=True)[()]
                self.episode = to_load["episode"]
                self.total_steps = to_load["total_steps"]
                self._q_parameters = to_load["q_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")

    def save_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "q_parameters": self._q_parameters,
            }
            np.save(checkpoint, to_save)
            print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                               self.total_steps,
                                                                                 checkpoint))
    def planning_update(
            self,
            timestep: dm_env.TimeStep,
            prev_timestep=None
    ) -> None:
        pass

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        pass

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        features = self._get_features([timestep.observation])
        next_features = self._get_features([new_timestep.observation])

        # self._replay.add([
        #     features,
        #     action,
        #     new_timestep.reward,
        #     new_timestep.discount,
        #     next_features,
        # ])

    def _log_summaries(self, losses_and_grads, summary_name):
        if self._logs is not None:
            losses = losses_and_grads["losses"]
            gradients = losses_and_grads["gradients"]
            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k),
                                      losses[k], step=ep)
                for k, v in gradients.items():
                    tf.summary.scalar("train/gradients/{}/{}".format(summary_name, k),
                                      gradients[k], step=ep)
                self.writer.flush()

    def get_values_for_all_states(self, all_states):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        return np.array(self._q_forward(self._q_parameters, np.asarray(features)), np.float)

        # losses = losses_and_grads["losses"]
        # gradients = losses_and_grads["gradients"]
        #
        # if self.episode % self._log_period == 0:
        #     for k, v in losses.items():
        #         tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), losses[k], step=self.episode)
        #     for k, v in gradients.items():
        #         tf.summary.scalar("train/gradients/{}/{}".format(summary_name, k), gradients[k], step=self.episode)
        #     self.writer.flush()

    def update_hyper_params(self, episode, total_episodes):
        # decay_period, step, warmup_steps, epsilon):
        """Returns the current epsilon for the agent's epsilon-greedy policy.
        This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
        al., 2015). The schedule is as follows:
          Begin at 1. until warmup_steps steps have been taken; then
          Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
          Use epsilon from there on.
        Args:
          decay_period: float, the period over which epsilon is decayed.
          step: int, the number of training steps completed so far.
          warmup_steps: int, the number of steps taken before epsilon is decayed.
          epsilon: float, the final value to which to decay the epsilon parameter.
        Returns:
          A float, the current epsilon value computed according to the schedule.
        """
        steps_left = total_episodes + 0 - episode
        bonus = (self._initial_epsilon - self._final_epsilon) * steps_left / total_episodes
        bonus = np.clip(bonus, 0., self._initial_epsilon - self._final_epsilon)
        self._epsilon = self._final_epsilon + bonus
        if self._logs is not None:
            # if self._max_len == -1:
            ep = self.total_steps
            # else:
            #     ep = self.episode
            if ep % self._log_period == 0:
                tf.summary.scalar("train/epsilon",
                                  self._epsilon, step=ep)
                self.writer.flush()

    # def td_error(self,
    #             transitions):
    #     o_tm1, a_tm1, r_t, d_t, o_t = transitions
    #     q_tm1 = self._q_forward(self._q_parameters, o_tm1)
    #     q_t = self._q_forward(self._q_parameters, o_t)
    #     q_target = r_t + d_t * self._discount * jnp.max(q_t, axis=-1)
    #     q_a_tm1 = jax.vmap(lambda q, a: q[a])(q_tm1, a_tm1)
    #
    #     td_error = q_target - q_a_tm1
    #     return td_error
