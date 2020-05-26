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

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class VanillaQT(Agent):
    def __init__(
            self,
            run_mode: str,
            action_spec: specs.DiscreteArray,
            network,
            batch_size: int,
            input_dim,
            discount: float,
            lr: float,
            max_len: int,
            lr_model: float,
            nrng,
            top_n: int,
            rng_seq,
            log_period: int,
            exploration_decay_period: int,
            seed: int = None,
            latent=False,
            target_networks=False,
            logs: str = "logs",
            feature_coder=None,
    ):
        super().__init__()

        self._run_mode = run_mode
        self._nA = action_spec.num_values
        self._discount = discount
        self._batch_size = batch_size
        self._input_dim = input_dim
        self._n = 0
        self._top_n = top_n
        self._run_mode = "{}".format(self._run_mode)
        self._max_len = max_len
        self._final_epsilon = 0.0
        self._initial_epsilon = 0.5
        self._epsilon = 0.5
        self._exploration_decay_period = exploration_decay_period
        self._nrng = nrng
        self._replay = Replay(capacity=1000, nrng=self._nrng)
        self._initial_lr = lr
        self._lr = lr
        self._lr_model = lr_model
        self._initial_lr_model = lr_model
        self._warmup_steps = 0
        self._logs = logs
        self._seed = seed
        self._max_len = max_len
        self._log_period = log_period

        if self._run_mode == "q":
            self._create_summary_dirs()

        def cross_entropy(logprobs, targets):
            target_class = np.argmax(targets, axis=-1)
            nll = np.take_along_axis(logprobs, np.expand_dims(target_class, axis=0), axis=0)
            ce = -np.mean(nll)
            return ce

        def log_softmax(x, axis=-1):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return np.log(e_x / e_x.sum(axis=axis, keepdims=True))

        def softmax(x, axis=-1):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / e_x.sum(axis=axis, keepdims=True)

        self._softmax = softmax
        self._log_softmax = log_softmax
        self._ce = cross_entropy
        self._nrng = nrng

        # Internalize the networks.
        self._q_network = network["qvalue"]["net"]
        self._q_parameters = network["qvalue"]["params"]

        self._network = network

        def q_loss(q_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = q_params[o_tm1, a_tm1]
            q_t = q_params[o_t]
            q_target = r_t + d_t * discount * np.max(q_t, axis=-1)
            td_error = (q_target - q_tm1)
            return np.mean(td_error ** 2), td_error

        self._q_loss_grad = q_loss
        self._q_opt_update = lambda gradient, params: np.add(params, self._lr * gradient)
        self._q_planning_opt_update = lambda gradient, params: np.add(params, self._lr * gradient)

    def _create_summary_dirs(self):
        if self._logs is not None:
            self._checkpoint_dir = os.path.join(self._logs,
                                            '{}/checkpoints/seed_{}'.format(self._run_mode, self._seed))
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)

            self._checkpoint_filename = "checkpoint.npy"

            self.writer = tf.summary.create_file_writer(
                os.path.join(self._logs, '{}/summaries/seed_{}'.format(self._run_mode, self._seed)))

            self._images_dir = os.path.join(self._logs, '{}/images/seed_{}'.format(self._run_mode, self._seed))
            if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        if not eval and self._nrng.rand() < self._epsilon:
            return self._nrng.randint(self._nA)
        q_values = self._q_network[timestep.observation]
        a = self._nrng.choice(np.flatnonzero(q_values == q_values.max()))
        return a

    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        o_tm1 = np.array([timestep.observation])
        a_tm1 = np.array([action])
        r_t = np.array([new_timestep.reward])
        d_t = np.array([new_timestep.discount])
        o_t = np.array([new_timestep.observation])
        transitions = [o_tm1, a_tm1, r_t, d_t, o_t]

        loss, gradient = self._q_loss_grad(self._q_network, transitions)
        self._q_network[o_tm1, a_tm1] = self._q_opt_update(gradient,
                                        self._q_network[o_tm1, a_tm1])

        losses_and_grads = {"losses": {"loss_q": np.array(loss)},
                            }
        self._log_summaries(losses_and_grads, "value")

    def model_based_train(self):
        return False

    def model_free_train(self):
        return True

    def load_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            if os.path.exists(checkpoint):
                to_load = np.load(checkpoint, allow_pickle=True)[()]
                self.episode = to_load["episode"]
                self.total_steps = to_load["total_steps"]
                self._v_network = to_load["v_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")

    def save_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "v_parameters": self._v_network,
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
        pass

    def _log_summaries(self, losses_and_grads, summary_name):
        if self._logs is not None:
            losses = losses_and_grads["losses"]

            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), np.array(v),
                                      step=ep)
                self.writer.flush()

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

    def get_values_for_all_states(self, all_states):
        return np.max(self._q_network[all_states], -1)

    def get_qvalues_for_all_states(self, all_states):
        return self._q_network[all_states]

    def get_policy_for_all_states(self, all_states):
        actions = np.argmax(self._q_network[all_states], axis=-1)

        return np.array(actions)