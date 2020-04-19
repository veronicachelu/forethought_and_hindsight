import numpy as np
import os
import jax.random as jrandom
import haiku as hk
import tensorflow as tf
from control_agents import VanillaQ
from network import *
import contextlib

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class GymSolver:
    def __init__(self, env,
                 input_dim, nA, space, aux_agent_configs, nrng):
        self._seed = 0
        self._env = env
        self._nA = nA
        self._nrng = nrng
        self._input_dim = input_dim
        self._nS = np.prod(input_dim)
        self._space = space
        self._num_episodes_training = 100
        self._aux_agent_configs = aux_agent_configs
        rng = jrandom.PRNGKey(seed=self._seed)
        self._rng, self._rng_target = jrandom.split(rng, 2)
        self._agent = self.get_q_learning_agent()
        self._policy = None
        self._assigned_pi = False
        self._v = None

    def get_q_learning_agent(self):
        network = get_network(
            pg=self._space["agent_config"]["pg"],
            num_hidden_layers=self._space["agent_config"]["num_hidden_layers"],
            num_units=self._space["agent_config"]["num_units"],
            nA=self._nA,
            input_dim=self._input_dim,
            rng=self._rng,
            rng_target=self._rng_target,
            feature_coder=self._space["env_config"]["feature_coder"],
            latent=self._space["agent_config"]["latent"],
            model_family="q",
            model_class=self._space["env_config"]["model_class"],
            target_networks=self._space["agent_config"]["target_networks"])

        agent = VanillaQ(run_mode='q',
            action_spec=self._env.action_spec(),
            network=network,
            batch_size=1,
            discount=0.99,
            lr=self._space["env_config"]["lr_q"],
            exploration_decay_period=self._num_episodes_training,
            nrng=self._nrng,
            seed=self._seed,
            logs=self._space["logs"],
            log_period=self._space["log_period"],
            latent=self._space["agent_config"]["latent"],
            feature_coder=self._space["env_config"]["feature_coder"],
            target_networks=self._space["agent_config"]["target_networks"]
        )
        return agent

    def train_agent(self, agent, environment, num_episodes, max_len):
        cumulative_reward = 0
        agent.load_model()
        ep_steps = []
        ep_rewards = []
        with agent.writer.as_default() if agent._logs is not None else dummy_context_mgr():
            for episode in np.arange(start=agent.episode, stop=num_episodes):
                # Run an episode.
                rewards = 0
                ep_reward = 0
                timestep = environment.reset()
                agent.update_hyper_params(episode, num_episodes)
                for t in range(max_len):

                    action = agent.policy(timestep)
                    new_timestep = environment.step(action)

                    agent.value_update(timestep, action, new_timestep)

                    rewards += new_timestep.reward
                    ep_reward += new_timestep.reward
                    if new_timestep.last():
                        break

                    timestep = new_timestep

                    agent.total_steps += 1

                cumulative_reward += rewards
                agent.episode += 1

                ep_steps.append(t)
                ep_rewards.append(ep_reward)

                if agent._logs is not None:
                    tf.summary.scalar("train/avg_reward", np.mean(ep_rewards), step=agent.episode)
                    tf.summary.scalar("train/avg_steps", np.mean(ep_steps), step=agent.episode)
                    agent.writer.flush()

        agent.save_model()

        return np.mean(ep_steps), np.mean(ep_rewards)

    def test_agent(self, agent, environment, num_episodes, max_len):
        cumulative_reward = 0
        ep_steps = []
        ep_rewards = []
        for episode in np.arange(start=0, stop=num_episodes):
            # Run an episode.
            rewards = 0
            ep_reward = 0
            timestep = environment.reset()
            for t in range(max_len):

                if not agent:
                    action = environment.action_spec().generate_value()
                else:
                    action = agent.policy(timestep)
                new_timestep = environment.step(action)

                rewards += new_timestep.reward
                ep_reward += new_timestep.reward
                if new_timestep.last():
                    break

                timestep = new_timestep

            cumulative_reward += rewards

            ep_steps.append(t)
            ep_rewards.append(ep_reward)

        return np.mean(ep_steps), np.mean(ep_rewards)


    def get_optimal_policy(self):
        if self._policy is None or self._assigned_pi is False:
            self.train_agent(self._agent, self._env,
                             self._num_episodes_training,
                             self._aux_agent_configs["max_len"])
            # print(self.test_agent(self._agent, self._env,
            #                  self._num_episodes_training,
            #                  self._aux_agent_configs["max_len"]))
            # print(self.test_agent(None, self._env,
            #                       self._num_episodes_training,
            #                       self._aux_agent_configs["max_len"]))
            self._assigned_pi = True
            self._policy = self._agent.policy

        return self._policy

    def get_estimated_v(self):
        if self._v is None or self._assigned_v is False:
            self.train_agent(self._agent, self._env,
                             self._num_episodes_training,
                             self._aux_agent_configs["max_len"])

            self._assigned_v = True
            self._v = self._agent.v_function

        return self._v

    def get_value_for_state(self, state):
        return self._v(state)

