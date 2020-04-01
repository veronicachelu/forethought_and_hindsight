import os
from typing import Any
from typing import Callable, Sequence, Tuple

import dm_env
import numpy as np
import tensorflow as tf
from dm_env import specs
from jax import numpy as jnp

from control_agents.tabular.tabular_control import TabularControl
from utils.replay import Replay

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class VanillaTabularControl(TabularControl):
    def __init__(
            self,
            run_mode: str,
            policy: specs.DiscreteArray,
            action_spec: specs.DiscreteArray,
            q_network: Network,
            model_network: Network,
            q_parameters: NetworkParameters,
            model_parameters: NetworkParameters,
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
            epsilon: float,
            log_period: int,
            rng: Tuple,
            nrng,
            input_dim: int,
            exploration_decay_period: int,
            seed: int = None,
            logs: str = "logs",
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

        if self._n != 0:
            self._run_mode = "{}_n{}".format(self._run_mode, self._n)

        self._epsilon = epsilon
        self._exploration_decay_period = exploration_decay_period
        self._nrng = nrng
        self._replay = Replay(capacity=replay_capacity, nrng=self._nrng)
        self._min_replay_size = min_replay_size
        self._lr = lr
        self._lr_model = lr_model
        self._lr_planning = lr_planning
        self._warmup_steps = 0
        self._logs = logs
        self._seed = seed
        self._max_len = max_len
        self._input_dim = input_dim
        self._log_period = log_period
        self._checkpoint_dir = os.path.join(self._logs,
                                        '{}/checkpoints/seed_{}'.format(self._run_mode, self._seed))
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self._checkpoint_filename = "checkpoint.npy"

        self._rng = rng
        self._nrng = nrng

        self.writer = tf.summary.create_file_writer(
            os.path.join(self._logs, '{}/summaries/seed_{}'.format(self._run_mode, seed)))

        self._images_dir = os.path.join(self._logs, '{}/images/seed_{}'.format(self._run_mode, seed))
        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        self._q_network = q_network

        self._o_network = model_network[0]
        self._fw_o_network = model_network[1]
        self._r_network = model_network[2]
        self._d_network = model_network[3]

        def q_loss(q_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = q_params[o_tm1]
            q_t = q_network[o_t]
            q_target = r_t + d_t * discount * np.max(q_t, axis=-1)
            td_error = (q_target - q_tm1)
            return np.mean(td_error ** 2), td_error

        self._q_loss_grad = q_loss
        self._q_opt_update = lambda gradient, params: np.add(params, self._lr * gradient)
        self._q_planning_opt_update = lambda gradient, params: np.add(params, self._lr_planning * gradient)

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        # Epsilon-greedy policy if not test policy.
        if not eval and self._nrng.rand() < self._epsilon:
            return self._nrng.randint(self._nA)
        q_values = self._q_network[timestep.observation]
        a = self._nrng.choice(np.flatnonzero(q_values == np.max(q_values)))
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
        self._q_network[o_tm1] = self._q_opt_update(gradient, self._q_network[o_tm1])

        losses_and_grads = {"losses": {"loss_q": np.array(loss)},
                            }
        self._log_summaries(losses_and_grads, "value")


    def model_based_train(self):
        return False

    def model_free_train(self):
        return True

    def load_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._q_network = to_load["q_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "q_parameters": self._q_network,
        }
        np.save(checkpoint, to_save)
        print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))
    def planning_update(
            self,
            timestep: dm_env.TimeStep,
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

    def update_hyper_params(self, step, total_steps):
        pass

