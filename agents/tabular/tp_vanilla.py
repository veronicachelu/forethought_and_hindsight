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


class TpVanilla(Agent):
    def __init__(
            self,
            run_mode: str,
            policy,
            action_spec: specs.DiscreteArray,
            network,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            model_learning_period: int,
            planning_iter: int,
            planning_period: int,
            planning_depth: int,
            lr: float,
            max_len: int,
            lr_model: float,
            lr_planning: float,
            log_period: int,
            nrng,
            rng,
            input_dim: int,
            exploration_decay_period: int,
            seed: int = None,
            latent=False,
            target_networks=False,
            logs: str = "logs",
            policy_type=None,
            feature_coder=None,
            # double_input_reward_model=False
    ):
        super().__init__()

        self._run_mode = run_mode
        self._pi = policy
        self._nA = action_spec.num_values
        self._discount = discount
        self._batch_size = batch_size
        self._model_learning_period = model_learning_period
        self._planning_iter = planning_iter
        self._planning_period = planning_period
        self._n = planning_depth
        self._replay_capacity = replay_capacity
        # self._double_input_reward_model = double_input_reward_model
        self._run_mode = "{}_{}_{}".format(self._run_mode, self._n, self._replay_capacity)

        self._exploration_decay_period = exploration_decay_period
        self._nrng = nrng

        self._replay = Replay(capacity=replay_capacity, nrng=self._nrng)
        self._min_replay_size = min_replay_size
        self._initial_lr = lr
        self._lr = lr
        self._lr_model = lr_model
        self._initial_lr_model = lr_model
        self._lr_planning = lr_planning
        self._initial_lr_planning = lr_planning
        self._warmup_steps = 0
        self._logs = logs
        self._seed = seed
        self._max_len = max_len
        self._input_dim = input_dim
        self._log_period = log_period

        if self._logs is not None:
            self._checkpoint_dir = os.path.join(self._logs,
                                            '{}_{}/checkpoints/seed_{}'.format(self._run_mode, self._lr_model, self._seed))
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)

            self._checkpoint_filename = "checkpoint.npy"

            self._nrng = nrng

            self.writer = tf.summary.create_file_writer(
                os.path.join(self._logs, '{}_{}/summaries/seed_{}'.format(self._run_mode, self._lr_model, seed)))

            self._images_dir = os.path.join(self._logs, '{}_{}/images/seed_{}'.format(self._run_mode, self._lr_model, seed))
            if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)

        # Internalize the networks.
        self._v_network = network["value"]["net"]
        self._v_parameters = network["value"]["params"]

        self._network = network

        def v_loss(v_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            v_tm1 = v_params[o_tm1]
            v_t = v_params[o_t]
            v_target = r_t + d_t * discount * v_t
            td_error = (v_target - v_tm1)
            return np.mean(td_error ** 2), td_error

        self._v_loss_grad = v_loss
        self._v_opt_update = lambda gradient, params: np.add(params, self._lr * gradient)
        self._v_planning_opt_update = lambda gradient, params: np.add(params, self._lr_planning * gradient)

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        return self._pi(timestep.observation, self._nrng)

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

        loss, gradient = self._v_loss_grad(self._v_network, transitions)
        self._v_network[o_tm1] = self._v_opt_update(gradient, self._v_network[o_tm1])

        losses_and_grads = {"losses": {"loss_v": np.array(loss)},
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
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k), losses[k],
                                      step=ep)
                self.writer.flush()

    def update_hyper_params(self, episode, total_episodes):
        pass
        # warmup_episodes = 0
        # flat_period = 0
        # decay_period = total_episodes - warmup_episodes - flat_period
        # if episode > warmup_episodes:
        #     steps_left = total_episodes - episode - flat_period
        #     if steps_left <= 0:
        #         return
        #     self._lr = self._initial_lr * (steps_left / decay_period)

