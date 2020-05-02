from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs
from copy import deepcopy
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

def td_learning(
    v_tm1,
    r_t,
    discount_t,
    v_t):

  target_tm1 = r_t + discount_t * jax.lax.stop_gradient(v_t)
  return target_tm1 - v_tm1

class LpIntrinsicVanilla(Agent):
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
            policy_type=None,
            target_networks=False,
            feature_coder=None,
            logs: str = "logs",
    ):
        super().__init__()

        self._run_mode = run_mode
        self._pi = policy
        self._policy_type = policy_type
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
        self._target_networks = target_networks

        self._nrng = nrng

        if feature_coder is not None:
            self._feature_mapper = FeatureMapper(feature_coder)
            sparse_feature_coder = deepcopy(feature_coder)
            sparse_feature_coder["type"] = "tile"
            self._sparse_feature_mapper = FeatureMapper(sparse_feature_coder)
        else:
            self._feature_mapper = None
            self._sparse_feature_mapper = None

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

        def v_loss(v_params, h_params, transitions):
            o_tm1, _, r_t, d_t, o_t = transitions
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            h_tm1 = lax.stop_gradient(self._h_network(h_params, o_tm1)) if self._latent else o_tm1
            v_tm1 = jnp.squeeze(self._v_network(v_params, h_tm1), axis=-1)
            v_t = jnp.squeeze(self._v_network(v_params, h_t), axis=-1)
            td_error = jax.vmap(rlax.td_learning)(v_tm1, r_t, d_t * discount, v_t)
            return jnp.mean(td_error ** 2)

        # Internalize the networks.
        self._v_network = network["value"]["net"]
        self._v_parameters = network["value"]["params"]

        self._network = network

        # self._v_step_schedule = optimizers.polynomial_decay(self._lr, self._exploration_decay_period, 0, 1)

        # if self._target_networks:
        #     self._target_v_network = network["target_value"]["net"]
        #     self._target_v_parameters = network["target_value"]["params"]
        #
        #     self._target_h_network = network["target_model"]["net"][0]
        #     self._target_o_network = network["target_model"]["net"][1]
        #     self._target_fw_o_network = network["target_model"]["net"][2]
        #     self._target_r_network = network["target_model"]["net"][3]
        #
        #     self._target_h_parameters = network["target_model"]["params"][0]
        #     self._target_o_parameters = network["target_model"]["params"][1]
        #     self._target_fw_o_parameters = network["target_model"]["params"][2]
        #     self._target_r_parameters = network["target_model"]["params"][3]
        #
        #     # self._planning_v_network = self._v_network #network["planning_value"]["net"]
        #     # self._planning_v_parameters = self._v_parameters #network["planning_value"]["params"]
        #     # Make an Adam optimizer.
        #     # pv_opt_init, pv_opt_update, pv_get_params = optimizers.adam(step_size=self._lr)
        #     # self._pv_opt_update = jax.jit(pv_opt_update)
        #     # self._pv_opt_init = pv_opt_init
        #     # self._pv_opt_state = pv_opt_init(self._planning_v_parameters)
        #     # self._pv_get_params = pv_get_params

        self._h_network = network["model"]["net"][0]
        self._h_parameters = network["model"]["params"][0]

        # This function computes dL/dTheta
        dwrt = [0, 1] if self._latent else 0
        self._v_loss_grad = jax.jit(jax.value_and_grad(v_loss, dwrt))

        # self._v_loss_grad = jax.value_and_grad(v_loss, 0)
        self._v_forward = jax.jit(self._v_network)
        self._h_forward = jax.jit(self._h_network)

        # Make an Adam optimizer.
        v_opt_init, v_opt_update, v_get_params = optimizers.adam(step_size=self._lr)
        self._v_opt_update = jax.jit(v_opt_update)
        value_params = [self._v_parameters, self._h_parameters] if self._latent else self._v_parameters
        self._v_opt_state = v_opt_init(value_params)
        self._v_get_params = v_get_params

    def _get_features(self, o):
        if self._feature_mapper is not None:
            return self._feature_mapper.get_features(o)
        else:
            return o

    def _get_sparse_features(self, o):
        if self._sparse_feature_mapper is not None:
            return self._sparse_feature_mapper.get_features(o)
        else:
            return o

    def policy(self,
               timestep: dm_env.TimeStep,
               eval: bool = False
               ) -> int:
        if self._policy_type == "estimated":
            return self._pi(timestep, eval=True)
        elif self._policy_type == "random":
            return self._pi(self._nrng)
        elif self._policy_type == "greedy":
            features = self._get_features(timestep.observation[None, ...])
            return self._pi(np.argmax(features[0]), self._nrng)
        elif self._policy_type == "continuous_greedy":
            features = self._get_sparse_features(timestep.observation[None, ...])
            return self._pi(np.argmax(features[0]), self._nrng)

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

        loss, gradients = self._v_loss_grad(self._v_parameters,
                                           self._h_parameters,
                                            transitions)
        if self._latent:
            gradients = list(gradients)
        self._v_opt_state = self._v_opt_update(self.episode, gradients,
                                               self._v_opt_state)
        value_params = self._v_get_params(self._v_opt_state)
        if self._latent:
            self._v_parameters, _ = value_params
            # self._v_parameters, self._h_parameters = value_params
        else:
            self._v_parameters = value_params
        losses_and_grads = {"losses": {"loss_v": np.array(loss)},}
                            # "v_tmn": self._v_forward(self._v_parameters, np.array(features))[0]}

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

    def load_m(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self._r_parameters = to_load["r_parameters"]
            self._o_parameters = to_load["o_parameters"]
            # self._v_parameters = to_load["v_parameters"]
            # self._planning_v_parameters = self._v_parameters
            # self._pv_opt_state = self._pv_opt_init(self._planning_v_parameters)
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_m(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "r_parameters": self._r_parameters,
            "o_parameters": self._o_parameters,
            "v_parameters": self._v_parameters,
        }
        np.save(checkpoint, to_save)
        print("Saved model for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))

    def load_v(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self._v_parameters = to_load["v_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_v(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "v_parameters": self._v_parameters,
        }
        np.save(checkpoint, to_save)
        print("Saved model for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))


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
        # return
        if self._logs is not None:
            losses = losses_and_grads["losses"]
            # gradients = losses_and_grads["gradients"]
            if self._max_len == -1:
                ep = self.total_steps
            else:
                ep = self.episode
            if ep % self._log_period == 0:
                for k, v in losses.items():
                    tf.summary.scalar("train/losses/{}/{}".format(summary_name, k),
                                      losses[k], step=ep)
                # for k, v in gradients.items():
                #     tf.summary.scalar("train/gradients/{}/{}".format(summary_name, k),
                #                       gradients[k], step=ep)
                self.writer.flush()

    def get_value_for_state(self, state):
        features = self._get_features(state[None, ...]) if self._feature_mapper is not None else state[None, ...]
        return np.squeeze(self._v_forward(self._v_parameters, features), axis=-1)[0]

    def get_values_for_all_states(self, all_states, ls=None):
        features = self._get_features(all_states) if self._feature_mapper is not None else all_states
        latents = self._h_forward(self._h_parameters, np.array(features)) if self._latent else features
        return np.array(np.squeeze(self._v_forward(self._v_parameters, np.asarray(latents, np.float)), axis=-1), np.float)

    def update_hyper_params(self, step, total_steps):
        pass
