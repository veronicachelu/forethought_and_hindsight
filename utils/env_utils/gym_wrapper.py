from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Tuple, Union
from basis.rbf import RBF
import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np
from control_agents import *
import jax.random as jrandom
import haiku as hk
from network import *
from utils.visualizer import plot_policy
# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]

class DMEnvFromGym(dm_env.Environment):
  def __init__(self, game):
    gym_env = gym.make(game)
    self.gym_env = gym_env
    # Convert gym action and observation spaces to dm_env specs.
    self._observation_spec = space2spec(self.gym_env.observation_space,
                                        name='observations')
    self._action_spec = space2spec(self.gym_env.action_space, name='actions')
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    observation = self.gym_env.reset()
    return dm_env.restart(observation)

  def step(self, action: int) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()

    # Convert the gym step result to a dm_env TimeStep.
    observation, reward, done, info = self.gym_env.step(action)
    self._reset_next_step = done

    if done:
      is_truncated = info.get('TimeLimit.truncated', False)
      if is_truncated:
        return dm_env.truncation(reward, observation)
      else:
        return dm_env.termination(reward, observation)
    else:
      return dm_env.transition(reward, observation)

  def close(self):
    self.gym_env.close()

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec

def space2spec(space: gym.Space, name: str = None):
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=space.low, maximum=space.high, name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                              maximum=1.0, name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=np.zeros(space.shape),
                              maximum=space.nvec, name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(space2spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {key: space2spec(value, name) for key, value in space.spaces.items()}

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


def train_agent(agent, environment, num_episodes, max_len):
    cumulative_reward = 0
    for episode in np.arange(start=agent.episode, stop=num_episodes):
        # Run an episode.
        rewards = 0
        timestep = env.reset()
        agent.update_hyper_params(episode, num_episodes)
        for t in range(max_len):

            action = agent.policy(timestep)
            new_timestep = environment.step(action)

            agent.value_update(timestep, action, new_timestep)

            rewards += new_timestep.reward

            if new_timestep.last():
                break

            timestep = new_timestep

            agent.total_steps += 1

        cumulative_reward += rewards
        agent.episode += 1

    return cumulative_reward


def test_agent(agent, environment, num_episodes, max_len):
    cumulative_reward = 0
    for episode in np.arange(start=0, stop=num_episodes):
        # Run an episode.
        rewards = 0
        timestep = env.reset()
        for t in range(max_len):
            if not agent:
                action = env.action_spec().generate_value()

            new_timestep = environment.step(action)

            rewards += new_timestep.reward

            if new_timestep.last():
                break
            timestep = new_timestep

        cumulative_reward += rewards

    return cumulative_reward

if __name__ == "__main__":
    # nrng = np.random.RandomState(0)
    # nS = 5
    # nA = 1
    # discount = 0.95
    # env = Shortcut(rng=nrng, obs_type="tabular", nS=nS, right_reward=1)
    # mdp_solver = ChainSolver(env, nS, nA, discount)
    # # policy = mdp_solver.get_optimal_policy()
    # v = mdp_solver.get_optimal_v()
    # v = env.reshape_v(v)
    # # plot_v(env, v, logs, env_type=FLAGS.env_type)
    # # plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
    # # eta_pi = mdp_solver.get_eta_pi(policy)
    # # plot_eta_pi(env, env.reshape_v(eta_pi)
    #
    game = 'CartPole-v0'
    env = DMEnvFromGym(game)
    observation_spec = env.observation_spec()
    rbf_coder = RBF()
    print(observation_spec.shape)
    action_spec = env.action_spec()
    print(action_spec.num_values)
    input_dim = env.observation_spec().shape
    nrng = np.random.RandomState(0)
    rng = jrandom.PRNGKey(seed=0)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    rng_sequence = hk.PRNGSequence(rng_agent)
    model_class = "linear"

    network = get_network(
        pg=False,
        num_hidden_layers=0,
        num_units=0,
        nA=action_spec.num_values,
        input_dim=input_dim,
        rng=rng_model,
        rng_target=rng_q,
        latent=False,
        model_family="q",
        model_class=model_class,
        target_networks=False)

    agent = VanillaQ(
        run_mode="q",
        policy=None,
        action_spec=env.action_spec(),
        network=network,
        batch_size=1,
        discount=0.95,
        replay_capacity=0,
        min_replay_size=1,
        model_learning_period=1,
        planning_iter=1,
        planning_period=1,
        planning_depth=0,
        lr=0.1,
        lr_model=0,
        lr_planning=0,
        exploration_decay_period=100,
        seed=0,
        nrng=nrng,
        rng=rng_agent,
        logs=None,
        max_len=100000,
        log_period=1,
        input_dim=observation_spec,
        latent=False,
        feature_mapper=rbf_coder,
        target_networks=False
        # double_input_reward_model=True
    )

    train_agent(agent, env, 100, 100)
    random_agent_reward = test_agent(None, env, 10, 100)
    print("Random agent reward {}".format(random_agent_reward / 10))
    q_agent_reward = test_agent(None, env, 10, 100)
    print("Q agent reward {}".format(q_agent_reward / 10))
    # q_agent_reward = 0
    # random_agent_reward = 0
    # for ep in range(100):
    #     ep_reward = 0
    #     ts = env.reset()
    #     while not ts.last():
    #         ts = env.step(agent.policy(ts, eval=True))
    #         print("reward {}".format(ts.reward))
    #         ep_reward += ts.reward
    #     print("discount {}".format(ts.discount))
    #     q_agent_reward += ep_reward

