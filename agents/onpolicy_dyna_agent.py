from typing import Any, Callable, Sequence
import os
from utils.replay import Replay

import dm_env
from dm_env import specs

import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import optimizers
from jax.experimental import stax
import numpy as np
from agents.dyna_agent import DynaAgent
import tensorflow as tf

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class OnPolicyDynaAgent(DynaAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(OnPolicyDynaAgent, self).__init__(**kwargs)

        def model_loss(online_params, transitions):
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            model_tm1 = self._model_network(online_params, o_tm1)
            # apply relu
            o_t = jax.vmap(lambda model, a: model[a][:-2])(model_tm1, a_tm1)
            # identity
            r_t = jax.vmap(lambda model, a: model[a][-2])(model_tm1, a_tm1)
            # apply sigmoid - but keep logits for loss
            d_t = jax.vmap(lambda model, a: jax.nn.sigmoid(model[a][-1]))(model_tm1, a_tm1)
            o_error = lax.stop_gradient(o_t_target) - o_t
            r_error = r_t - lax.stop_gradient(r_t_target)
            d_error = - jnp.log(d_t) * d_t_target - jnp.log(1 - d_t) * (1 - d_t_target)

            total_error = jnp.mean(o_error ** 2) + jnp.mean(r_error ** 2) + jnp.mean(d_error)
            return total_error

        def model_o_loss(online_params, transitions):
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            model_tm1 = self._model_network(online_params, o_tm1)

            o_t = jax.vmap(lambda model, a: model[a][:-2])(model_tm1, a_tm1)
            o_error = lax.stop_gradient(o_t_target) - o_t

            return jnp.mean(o_error ** 2)

        def model_r_loss(online_params, transitions):
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            model_tm1 = self._model_network(online_params, o_tm1)
            r_t = jax.vmap(lambda model, a: model[a][-2])(model_tm1, a_tm1)
            r_error = r_t - lax.stop_gradient(r_t_target)

            return jnp.mean(r_error ** 2)

        def model_d_loss(online_params, transitions):
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            model_tm1 = self._model_network(online_params, o_tm1)
            d_t = jax.vmap(lambda model, a: jax.nn.sigmoid(model[a][-1]))(model_tm1, a_tm1)
            d_error = - jnp.log(d_t) * d_t_target - jnp.log(1 - d_t) * (1 - d_t_target)

            return jnp.mean(d_error)

        self._model_o_loss = model_o_loss
        self._model_r_loss = model_r_loss
        self._model_d_loss = model_d_loss
        # This function computes dL/dTheta
        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss))
        self._model_forward = jax.jit(self._model_network)

        # Make an Adam optimizer.
        model_opt_init, model_opt_update, model_get_params = optimizers.adam(step_size=self._lr_model)
        self._model_opt_update = jax.jit(model_opt_update)
        self._model_opt_state = model_opt_init(self._model_parameters)
        self._model_get_params = model_get_params

    def model_update(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        if not self._reset_model_update:
            self._add_transition(timestep, action, new_timestep)

        if self._reset_model_update:
            transitions = self._create_transition_batch()

            loss, gradient = self._model_loss_grad(self._model_parameters,
                                    transitions)
            self._model_opt_state = self._model_opt_update(self.total_steps, gradient,
                                                   self._model_opt_state)
            self._model_parameters = self._model_get_params(self._model_opt_state)

            loss_o = np.array(self._model_o_loss(self._model_parameters, transitions))
            loss_r = np.array(self._model_r_loss(self._model_parameters, transitions))
            loss_d = np.array(self._model_d_loss(self._model_parameters, transitions))
            loss = np.array(loss)
            losses_and_grads = {"losses": {"loss_o": loss_o,
                                           "loss_r": loss_r,
                                           "loss_d": loss_d,
                                           "loss": loss
                                           },
                                "gradients": {
                                    "grad_norm": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))
                                              }
                                }
            self._log_summaries(losses_and_grads, "model")

    def planning_update(
            self
    ):
        if self._replay.size < self._min_replay_size:
            return
        if self.total_steps % self._planning_period == 0:
            for k in range(self._planning_iter):
                transitions = self._replay.sample(self._batch_size)
                # o_tm1, a_tm1, r_t_target, d_t_target, o_t_target
                o_tm1, a_tm1, r_t, d_t, o_t = transitions
                # r_t, d_t, o_t
                model_tm1 = self._model_forward(self._model_parameters, o_tm1)
                model_o_t = np.array(jax.vmap(lambda model, a: model[a][:-2])(model_tm1, a_tm1))
                model_r_t = np.array(jax.vmap(lambda model, a: model[a][-2])(model_tm1, a_tm1))
                model_d_t = np.array(jax.vmap(lambda model, a:
                                        random.bernoulli(self._rng,
                                                         p=jax.nn.sigmoid(model[a][-1])
                                                         ))(model_tm1, a_tm1), dtype=np.int32)
                decision_loss = np.array(jnp.mean(jnp.argmax(o_t, axis=-1) - jnp.argmax(model_o_t, axis=-1)))
                transitions[-1] = model_o_t
                transitions[-3] = model_r_t
                transitions[-2] = model_d_t
                # transitions.extend([r_t, d_t, o_t])
                # plan on batch of transitions
                loss, gradient = self._q_loss_grad(self._q_parameters,
                                        transitions)
                self._q_opt_state = self._q_opt_update(self.total_steps, gradient,
                                                       self._q_opt_state)
                self._q_parameters = self._q_get_params(self._q_opt_state)

                losses_and_grads = {"losses": {"loss_q_planning": np.array(loss),
                                                "loss_decision": decision_loss},
                                    "gradients": {"grad_norm_q_plannin": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
                self._log_summaries(losses_and_grads, "value_planning")

        # if self._replay.size < self._min_replay_size:
        #     return
        # for k in range(self._planning_iter):
        #     transitions = self._replay.sample(self._batch_size)
        #     # plan on batch of transitions
        #     loss, gradient = self._q_loss_grad(self._q_parameters,
        #                             transitions)
        #     self._q_opt_state = self._q_opt_update(self.total_steps, gradient,
        #                                            self._q_opt_state)
        #     self._q_parameters = self._q_get_params(self._q_opt_state)
        #
        #     losses_and_grads = {"losses": {"loss_q_planning": np.array(loss)},
        #                         "gradients": {"grad_norm_q_planning": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
        #     self._log_summaries(losses_and_grads, "value_planning")

    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        # Add this states and actions to replay.
        # self._replay.add([
        #     timestep.observation,
        #     action
        # ])
        # Add this transition to replay.
        self._replay.add([
            timestep.observation,
            action,
            new_timestep.reward,
            new_timestep.discount,
            new_timestep.observation,
        ])

    def load_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        if os.path.exists(checkpoint):
            to_load = np.load(checkpoint, allow_pickle=True)[()]
            self.episode = to_load["episode"]
            self.total_steps = to_load["total_steps"]
            self._q_parameters = to_load["q_parameters"]
            self._replay = to_load["replay"]
            self._model_parameters = to_load["model_parameters"]
            print("Restored from {}".format(checkpoint))
        else:
            print("Initializing from scratch.")

    def save_model(self):
        checkpoint = os.path.join(self._checkpoint_dir, self._checkpoint_filename)
        to_save = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "q_parameters": self._q_parameters,
            "replay": self._replay,
            "model_parameters": self._model_parameters,
        }
        np.save(checkpoint, to_save)
        print("Saved checkpoint for episode {}, total_steps {}: {}".format(self.episode,
                                                                           self.total_steps,
                                                                           checkpoint))

    def model_based_train(self):
        return True

    def model_free_train(self):
        return True

    def _add_transition(self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        self._model_update_batch["o_tm1"].append(timestep.observation)
        self._model_update_batch["a_tm1"].append(action)
        self._model_update_batch["o_t"].append(new_timestep.observation)
        self._model_update_batch["r_t"].append(new_timestep.reward)
        self._model_update_batch["d_t"].append(new_timestep.discount)
        if len(self._model_update_batch["o_tm1"]) == self._model_learning_period:
            self._reset_model_update = True

    def _create_transition_batch(self):
        o_tm1 = np.array(self._model_update_batch["o_tm1"])
        a_tm1 = np.array(self._model_update_batch["a_tm1"])
        o_t = np.array(self._model_update_batch["o_t"])
        r_t = np.array(self._model_update_batch["r_t"])
        d_t = np.array(self._model_update_batch["d_t"])

        transitions = [o_tm1, a_tm1, r_t, d_t, o_t]

        self._model_update_batch["o_tm1"].clear()
        self._model_update_batch["a_tm1"].clear()
        self._model_update_batch["o_t"].clear()
        self._model_update_batch["r_t"].clear()
        self._model_update_batch["d_t"].clear()
        self._reset_model_update = False
        return transitions