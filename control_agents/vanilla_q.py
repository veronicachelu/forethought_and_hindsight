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

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class VanillaQ(Agent):
    def __init__(
            self,
            run_mode: str,
            action_spec: specs.DiscreteArray,
            network,
            batch_size: int,
            discount: float,
            lr: float,
            log_period: int,
            nrng,
            rng_seq,
            exploration_decay_period: int,
            seed: int = None,
            latent=False,
            target_networks=False,
            feature_coder=None,
            logs: str = "logs",
    ):
        super().__init__()

        self._run_mode = run_mode
        self._nA = action_spec.num_values
        self._discount = discount
        self._batch_size = batch_size
        self._latent = latent
        # self._double_input_reward_model = double_input_reward_model
        self._run_mode = "{}".format(self._run_mode)

        self._epsilon = 0.1
        self._exploration_decay_period = exploration_decay_period
        self._nrng = nrng
        self._lr = lr
        self._warmup_steps = 0
        self._logs = logs
        self._seed = seed
        self._log_period = log_period

        if feature_coder is not None:
            self._feature_mapper = FeatureMapper(feature_coder)
        else:
            self._feature_mapper = None

        if self._logs is not None:
            self._checkpoint_dir = os.path.join(self._logs,
                                            '{}/checkpoints/seed_{}'.format(self._run_mode, self._seed))
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)

            self._checkpoint_filename = "checkpoint.npy"

            self.writer = tf.summary.create_file_writer(
                os.path.join(self._logs, '{}/summaries/seed_{}'.format(self._run_mode, seed)))

            self._images_dir = os.path.join(self._logs, '{}/images/seed_{}'.format(self._run_mode, seed))
            if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)

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
            return self._feature_mapper.get_features(o)
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
        self._q_opt_state = self._q_opt_update(self.total_steps, gradient,
                                               self._q_opt_state)
        self._q_parameters = self._q_get_params(self._q_opt_state)

        losses_and_grads = {"losses": {"loss_q": np.array(loss)},
                            "gradients": {"grad_norm_q": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
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
        pass

    def _log_summaries(self, losses_and_grads, summary_name):
        return

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
