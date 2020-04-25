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


class ACVanilla(Agent):
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
            rng,
            nrng,
            input_dim: int,
            exploration_decay_period: int,
            seed: int = None,
            latent=False,
            policy_type=None,
            target_networks=False,
            feature_coder=None,
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
        self._replay_capacity = replay_capacity
        self._latent = latent
        self._run_mode = "{}_{}_{}".format(self._run_mode, self._n, self._replay_capacity)

        self._exploration_decay_period = exploration_decay_period
        self._rng = rng
        self._nrng = nrng

        if feature_coder is not None:
            self._feature_mapper = FeatureMapper(feature_coder)
        else:
            self._feature_mapper = None

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
                                            '{}/checkpoints/seed_{}'.format(self._run_mode, self._seed))
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)

            self._checkpoint_filename = "checkpoint.npy"

            self._nrng = nrng

            self.writer = tf.summary.create_file_writer(
                os.path.join(self._logs, '{}/summaries/seed_{}'.format(self._run_mode, seed)))

            self._images_dir = os.path.join(self._logs, '{}/images/seed_{}'.format(self._run_mode, seed))
            if not os.path.exists(self._images_dir):
                os.makedirs(self._images_dir)

        def ac_loss(v_params, pi_params, h_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            h_tm1 = lax.stop_gradient(self._h_network(h_params, o_tm1)) if self._latent else o_tm1
            v_tm1 = self._v_network(v_params, h_tm1)
            v_t = self._v_network(v_params, h_t)
            td_error = jax.vmap(rlax.td_learning)(v_tm1, r_t, d_t * discount, v_t)
            critic_loss = jnp.mean(td_error ** 2)

            pi_tm1 = self._pi_network(pi_params, h_tm1)
            actor_loss = rlax.policy_gradient_loss(pi_tm1, a_tm1, td_error,
                                                             jnp.ones_like(td_error))

            total_loss = actor_loss + critic_loss
            return total_loss,\
                   {"critic": critic_loss,
                    "actor": actor_loss}

        # Internalize the networks.
        self._v_network = network["value"]["net"]
        self._v_parameters = network["value"]["params"]

        self._pi_network = network["pi"]["net"]
        self._pi_parameters = network["pi"]["params"]

        self._h_network = network["model"]["net"][0]
        self._o_network = network["model"]["net"][1]
        self._fw_o_network = network["model"]["net"][2]
        self._r_network = network["model"]["net"][3]

        self._h_parameters = network["model"]["params"][0]
        self._o_parameters = network["model"]["params"][1]
        self._fw_o_parameters = network["model"]["params"][2]
        self._r_parameters = network["model"]["params"][3]

        # This function computes dL/dTheta
        self._ac_loss_grad = jax.jit(jax.value_and_grad(ac_loss, [0, 1, 2], has_aux=True))
        self._v_forward = jax.jit(self._v_network)
        self._pi_forward = jax.jit(self._pi_network)

        # Make an Adam optimizer.
        ac_opt_init, ac_opt_update, ac_get_params = optimizers.adam(step_size=self._lr)
        self._ac_opt_update = jax.jit(ac_opt_update)
        self._ac_opt_state = ac_opt_init([self._v_parameters,
                                          self._pi_parameters,
                                          self._h_parameters])
        self._ac_get_params = ac_get_params

    def _get_features(self, o):
        if self._feature_mapper is not None:
            return self._feature_mapper.get_features(o)
        else:
            return o

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        key = next(self._rng)
        features = self._get_features(timestep.observation[None, ...])
        latent_state = self._pi_forward(self._pi_parameters, features)
        pi_logits = self._pi_forward(self._pi_parameters, latent_state)
        action = jax.random.categorical(key, pi_logits).squeeze()
        return int(action)

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

        (total_loss, losses), gradients = self._ac_loss_grad(self._v_parameters,
                                             self._pi_parameters,
                                             self._h_parameters,
                                            transitions)
        self._ac_opt_state = self._ac_opt_update(self.total_steps, list(gradients),
                                               self._ac_opt_state)
        ac_parameters = self._ac_get_params(self._ac_opt_state)
        self._v_parameters, self._pi_parameters, self._h_parameters = ac_parameters

        losses_and_grads = {"losses": {
                                    "total_loss": np.array(total_loss),
                                    "critic_loss": np.array(losses["critic"]),
                                    "actor_loss": np.array(losses["actor"])
                            },
                            "gradients": {
                                # "grad_norm_v":
                                #               np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2)
                                #                              for g in gradients]))}
                                }}
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
                self._v_parameters = to_load["v_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")

    def save_model(self):
        if self._logs is not None:
            return
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "v_parameters": self._v_parameters,
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
            gradients = losses_and_grads["gradients"]
            # if self._max_len == -1:
            ep = self.total_steps
            # else:
            #     ep = self.episode
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
        latents = self._h_forward(self._h_parameters, np.array(features)) if self._latent else features
        return np.array(self._v_forward(self._v_parameters, np.asarray(latents, np.float)), np.float)

    # def get_policy(self, all_states):
    #     latents = self._h_forward(self._h_parameters, np.array(all_states)) if self._latent else all_states
    #     return np.eye(self._nA)[np.argmax(np.array(self._pi_forward(self._pi_parameters, np.asarray(latents, np.float)), np.float), -1)]

    def update_hyper_params(self, step, total_steps):
        pass
