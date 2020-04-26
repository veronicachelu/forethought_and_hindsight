from typing import Any, Callable, Sequence
import os
from utils.replay import Replay
from typing import Callable, List, Mapping, Sequence, Text, Tuple, Union
import dm_env
from dm_env import specs
from rlax._src import distributions
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


class ACMB(Agent):
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
        self._run_mode = "{}".format(self._run_mode)

        self._exploration_decay_period = exploration_decay_period
        self._nrng = nrng
        self._rng_seq = rng_seq
        self._epsilon = 1.0
        self._sequence = []
        self._should_reset_sequence = False
        self._update_every = 50

        if feature_coder is not None:
            self._feature_mapper = FeatureMapper(feature_coder)
        else:
            self._feature_mapper = None

        self._lr = lr
        self._warmup_steps = 0
        self._logs = logs
        self._seed = seed
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

        def model_loss(o_online_params,
                       r_online_params,
                       transitions):
            o_tmn_target = transitions[0][0]
            o_t = transitions[-1][-1]
            model_o_tmn = self._o_network(o_online_params, o_t)

            o_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_o_tmn, o_tmn_target))

            r_input = jnp.concatenate([model_o_tmn, o_t], axis=-1)
            model_r_tmn = self._r_network(r_online_params, lax.stop_gradient(r_input))
            r_t_target = 0
            for i, t in enumerate(transitions):
                r_t_target += (self._discount ** i) * t[2]

            r_loss = jnp.mean(jax.vmap(rlax.l2_loss)(model_r_tmn, r_t_target))
            total_loss = o_loss + r_loss

            return total_loss, {"o_loss": o_loss,
                               "r_loss": r_loss
                               }

        def ac_loss(v_params, pi_params, h_params, transitions):
            a_tm1 = np.array([t[1] for t in transitions])
            o_tm1 = np.array([t[0] for t in transitions])
            o_t = np.array([t[-1] for t in transitions])
            r_t = np.array([t[2] for t in transitions])
            d_t = np.array([t[3] for t in transitions])
            h_t = lax.stop_gradient(self._h_network(h_params, o_t)) if self._latent else o_t
            h_tm1 = lax.stop_gradient(self._h_network(h_params, o_tm1)) if self._latent else o_tm1

            v_tm1 = self._v_network(v_params, h_tm1)
            v_t = self._v_network(v_params, h_t)
            td_error = jax.vmap(rlax.td_learning)(v_tm1, r_t, d_t * discount, v_t)
            critic_loss = jnp.mean(td_error ** 2)

            pi_tm1 = self._pi_network(pi_params, h_tm1)
            actor_loss = rlax.policy_gradient_loss(pi_tm1, a_tm1, td_error,
                                                   jnp.ones_like(td_error))

            entropy_error = distributions.softmax().entropy(pi_tm1)
            entropy = - jnp.mean(entropy_error)

            entropy_loss = - self._epsilon * entropy

            total_loss = actor_loss + critic_loss #+ entropy_loss
            return total_loss,\
                   {"critic": critic_loss,
                    "actor": actor_loss,
                    "entropy": entropy_loss
                    }

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
        self._ac_loss_grad = (jax.value_and_grad(ac_loss, [0, 1, 2], has_aux=True))
        # self._ac_loss_grad = jax.jit(jax.value_and_grad(ac_loss, [0, 1, 2], has_aux=True))
        self._v_forward = jax.jit(self._v_network)
        self._pi_forward = jax.jit(self._pi_network)

        # Make an Adam optimizer.
        self._step_schedule = optimizers.polynomial_decay(self._lr,
                                                                self._exploration_decay_period, 0, 0.9)
        ac_opt_init, ac_opt_update, ac_get_params = optimizers.adam(step_size=self._step_schedule)
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
        key = next(self._rng_seq)
        features = self._get_features(timestep.observation[None, ...])
        pi_logits = self._pi_forward(self._pi_parameters, features)
        action = jax.random.categorical(key, pi_logits).squeeze()
        return int(action)

    def value_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if len(self._sequence) >= self._update_every:
            (total_loss, losses), gradients = self._ac_loss_grad(self._v_parameters,
                                                 self._pi_parameters,
                                                 self._h_parameters,
                                                self._sequence)
            self._ac_opt_state = self._ac_opt_update(self.episode, list(gradients),
                                                   self._ac_opt_state)
            ac_parameters = self._ac_get_params(self._ac_opt_state)
            self._v_parameters, self._pi_parameters, self._h_parameters = ac_parameters

            losses_and_grads = {"losses": {
                                        "total_loss": np.array(total_loss),
                                        "critic_loss": np.array(losses["critic"]),
                                        "entropy_loss": np.array(losses["entropy"]),
                                        "actor_loss": np.array(losses["actor"])
                                },
                                "gradients": {
                                    }}
            self._log_summaries(losses_and_grads, "value")
            self._sequence = []

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
                self._pi_parameters = to_load["pi_parameters"]
                print("Restored from {}".format(checkpoint))
            else:
                print("Initializing from scratch.")

    def save_model(self):
        if self._logs is not None:
            checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
            to_save = {
                "episode": self.episode,
                "total_steps": self.total_steps,
                "v_parameters": self._v_parameters,
                "pi_parameters": self._pi_parameters,
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
        features = self._get_features([timestep.observation])[0]
        next_features = self._get_features([new_timestep.observation])[0]
        transitions = [features,
                       action,
                       new_timestep.reward,
                       new_timestep.discount,
                       next_features]

        self._sequence.append(transitions)

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

    def update_hyper_params(self, episode, total_episodes):
        steps_left = self._exploration_decay_period - episode
        bonus = (1.0 - self._epsilon) * steps_left / self._exploration_decay_period
        bonus = np.clip(bonus, 0., 1. - self._epsilon)
        self._epsilon = self._epsilon + bonus

