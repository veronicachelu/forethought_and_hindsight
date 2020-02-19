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
from agents.vanilla_agent import VanillaAgent
import tensorflow as tf

NetworkParameters = Sequence[Sequence[jnp.DeviceArray]]
Network = Callable[[NetworkParameters, Any], jnp.DeviceArray]


class DynaAgent(VanillaAgent):
    def __init__(
            self,
            **kwargs
    ):
        super(DynaAgent, self).__init__(**kwargs)

        def model_loss(online_params, transitions):
            o_tm1, a_tm1, r_t_target, d_t_target, o_t_target = transitions
            model_tm1 = self._model_network(online_params, o_tm1)
            o_t, r_t, d_t_logits = jax.vmap(lambda model, a:
                                     (model[a][:-3],
                                      model[a][-3],
                                      stax.logsoftmax(model[a][-2:])
                                      # jax.nn.sigmoid(model[a][-1]))
                                      ))(model_tm1, a_tm1)
            o_error = o_t - lax.stop_gradient(o_t_target)
            r_error = r_t - lax.stop_gradient(r_t_target)
            d_t_target = jnp.array(lax.stop_gradient(d_t_target), dtype=np.int32)
            # target_class = jnp.argmax(jnp.stack([d_t_target, 1 - d_t_target], axis=-1), axis=-1)
            nll = jnp.take_along_axis(d_t_logits, jnp.expand_dims(d_t_target, axis=-1), axis=1)
            # d_error = - jnp.log(d_t) * d_t_target - jnp.log(1 - d_t) * (1 - d_t_target)
            d_error = - jnp.mean(nll)
            total_error = jnp.mean(o_error ** 2) + jnp.mean(r_error ** 2) + d_error
            return total_error

        # This function computes dL/dTheta
        self._model_loss_grad = jax.jit(jax.value_and_grad(model_loss))
        self._model_forward = jax.jit(self._model_network)

        def q_planning_loss(q_params, model_params, transitions):
            o_tm1, a_tm1 = transitions
            model_tm1 = self._model_network(model_params, o_tm1)
            model_o_t, model_r_t, model_d_t = jax.vmap(lambda model, a:
                                                       (model[a][:-3],
                                                        model[a][-3],
                                                        jnp.argmax(model[a][-2:], axis=-1)
                                                        # random.bernoulli(self._rng,
                                                        #                  p=jax.nn.sigmoid(model[a][-1])))
                                                        ))(model_tm1, a_tm1)
            model_o_t, model_r_t, model_d_t = lax.stop_gradient(model_o_t),\
                                              lax.stop_gradient(model_r_t),\
                                              lax.stop_gradient(model_d_t)
            q_tm1 = self._q_network(q_params, o_tm1)
            q_t = self._q_network(q_params, model_o_t)
            q_target = model_r_t + model_d_t * self._discount * jnp.max(q_t, axis=-1)
            q_a_tm1 = jax.vmap(lambda q, a: q[a])(q_tm1, a_tm1)
            # td_error = lax.stop_gradient(q_target) - q_a_tm1
            td_error = q_a_tm1 - lax.stop_gradient(q_target)

            return jnp.mean(td_error ** 2)

        def debugging(model_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            model_tm1 = self._model_forward(model_params, o_tm1)
            model_o_t, model_r_t, model_d_t, model_d_logits = jax.vmap(lambda model, a:
                                                       (model[a][:-3],
                                                        model[a][-3],
                                                        jnp.argmax(model[a][-2:], axis=-1),
                                                        model[a][-2:]
                                                        # random.bernoulli(self._rng,
                                                        #                  p=jax.nn.sigmoid(model[a][-1])))
                                                        ))(model_tm1, a_tm1)
            model_o_t, model_r_t, model_d_t = lax.stop_gradient(model_o_t), \
                                              lax.stop_gradient(model_r_t), \
                                              lax.stop_gradient(model_d_t)
            d_t = jnp.array(d_t, dtype=np.int32)
            o_loss = jnp.mean((model_o_t - o_t) ** 2)
            reward_loss = jnp.mean((r_t - model_r_t) ** 2)
            d_decision_loss = jnp.mean((model_d_t - d_t) ** 2)
            d_loss = - jnp.mean(jnp.take_along_axis(model_d_logits, jnp.expand_dims(d_t, axis=-1), axis=-1))
            return o_loss, reward_loss, d_decision_loss, d_loss

        def debugging_gradients_o(model_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            model_tm1 = self._model_forward(model_params, o_tm1)
            model_o_t, model_r_t, model_d_t, model_d_logits = jax.vmap(lambda model, a:
                                                                       (model[a][:-3],
                                                                        model[a][-3],
                                                                        jnp.argmax(model[a][-2:], axis=-1),
                                                                        model[a][-2:]
                                                                        # random.bernoulli(self._rng,
                                                                        #                  p=jax.nn.sigmoid(model[a][-1])))
                                                                        ))(model_tm1, a_tm1)
            o_loss = jnp.mean((model_o_t - lax.stop_gradient(o_t)) ** 2)
            return o_loss

        def debugging_gradients_r(model_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            model_tm1 = self._model_forward(model_params, o_tm1)
            model_o_t, model_r_t, model_d_t, model_d_logits = jax.vmap(lambda model, a:
                                                                       (model[a][:-3],
                                                                        model[a][-3],
                                                                        jnp.argmax(model[a][-2:], axis=-1),
                                                                        model[a][-2:]
                                                                        # random.bernoulli(self._rng,
                                                                        #                  p=jax.nn.sigmoid(model[a][-1])))
                                                                        ))(model_tm1, a_tm1)
            reward_loss = jnp.mean((model_r_t - lax.stop_gradient(r_t)) ** 2)
            return reward_loss

        def debugging_gradients_d(model_params, transitions):
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            model_tm1 = self._model_forward(model_params, o_tm1)
            model_o_t, model_r_t, model_d_t, model_d_logits = jax.vmap(lambda model, a:
                                                                       (model[a][:-3],
                                                                        model[a][-3],
                                                                        jnp.argmax(model[a][-2:], axis=-1),
                                                                        model[a][-2:]
                                                                        # random.bernoulli(self._rng,
                                                                        #                  p=jax.nn.sigmoid(model[a][-1])))
                                                                        ))(model_tm1, a_tm1)
            d_t = jnp.array(d_t, dtype=np.int32)
            d_loss = - jnp.mean(jnp.take_along_axis(model_d_logits, jnp.expand_dims(d_t, axis=-1), axis=-1))
            return d_loss

        self._debugging = jax.jit(debugging)
        self._debugging_gradients_o = jax.jit(jax.grad(debugging_gradients_o))
        self._debugging_gradients_r = jax.jit(jax.grad(debugging_gradients_o))
        self._debugging_gradients_d = jax.jit(jax.grad(debugging_gradients_o))
        # This function computes dL/dTheta
        self._q_planning_loss_grad = jax.jit(jax.value_and_grad(q_planning_loss))

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
        transitions = [np.array([timestep.observation]),
                       np.array([action]),
                       np.array([new_timestep.reward]),
                       np.array([new_timestep.discount]),
                       np.array([new_timestep.observation])]

        loss, gradient = self._model_loss_grad(self._model_parameters,
                                transitions)
        self._model_opt_state = self._model_opt_update(self.total_steps, gradient,
                                               self._model_opt_state)
        self._model_parameters = self._model_get_params(self._model_opt_state)

        loss = np.array(loss)
        debugging_losses = self._debugging(self._model_parameters, transitions)
        debugging_gradients = []
        debugging_gradients.append(self._debugging_gradients_o(self._model_parameters, transitions))
        debugging_gradients.append(self._debugging_gradients_r(self._model_parameters, transitions))
        debugging_gradients.append(self._debugging_gradients_d(self._model_parameters, transitions))
        debugging_losses = list(debugging_losses)
        for i in range(len(debugging_losses)):
            debugging_losses[i] = np.array(debugging_losses[i])
        for i in range(len(debugging_gradients)):
            debugging_gradients[i] = np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in debugging_gradients[i]]))
        # "grad_norm": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))
        o_worst_case_loss, reward_loss, d_decision_loss, d_loss = debugging_losses
        grad_norm_o, grad_norm_r, grad_norm_d = debugging_gradients
        losses_and_grads = {"losses": {
                                       "loss": loss,
                                        "o_loss": o_worst_case_loss,
                                        "reward_loss": reward_loss,
                                        "d_decision_loss": d_decision_loss,
                                        "d_loss": d_loss
                                       },
                            "gradients": {
                                "grad_norm": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient])),
                                "grad_norm_o": grad_norm_o,
                                "grad_norm_r": grad_norm_r,
                                "grad_norm_d": grad_norm_d
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
                # plan on batch of transitions
                loss, gradient = self._q_planning_loss_grad(self._q_parameters,
                                                            self._model_parameters,
                                                            transitions[:2])
                self._q_opt_state = self._q_opt_update(self.total_steps, gradient,
                                                       self._q_opt_state)
                self._q_parameters = self._q_get_params(self._q_opt_state)


                losses_and_grads = {"losses": {"loss_q_planning": np.array(loss),
                                                },
                                    "gradients": {"grad_norm_q_plannin": np.sum(np.sum([np.linalg.norm(np.asarray(g), ord=2) for g in gradient]))}}
                self._log_summaries(losses_and_grads, "value_planning")


    def save_transition(
            self,
            timestep: dm_env.TimeStep,
            action: int,
            new_timestep: dm_env.TimeStep,
    ):
        # self._replay.add([
        #     timestep.observation,
        #     action
        # ])
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
