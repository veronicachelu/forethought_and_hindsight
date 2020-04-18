import dm_env
import tensorflow as tf

from agents import Agent
from utils.visualizer import *

def run_episodic(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        max_len: int,
        ):
    cumulative_reward = 0
    agent.load_model()
    ep_steps = []
    ep_rewards = []
    with agent.writer.as_default():
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

            tf.summary.scalar("train/avg_reward", np.mean(ep_rewards), step=agent.episode)
            tf.summary.scalar("train/avg_steps", np.mean(ep_steps), step=agent.episode)
            agent.writer.flush()

    agent.save_model()

    return np.mean(ep_steps), np.mean(ep_rewards)
