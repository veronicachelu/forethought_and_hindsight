from agents import Agent
import dm_env
import tensorflow as tf
import numpy as np

def run(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        num_test_episodes: int,
        max_len: int,
        log_period: 10,
        verbose: bool = False) -> None:

    agent.load_model()

    rewards_window = []
    t_window = []

    with agent.writer.as_default():
        for episode in np.arange(start=agent.episode, stop=num_episodes):
            # Run an episode.
            rewards = 0
            timestep = environment.reset()
            for t in range(max_len):

                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                if agent.model_based_train():
                    agent.save_transition(timestep, action, new_timestep)
                    agent.model_update(timestep, action, new_timestep)

                if agent.model_free_train():
                    agent.value_update(timestep, action, new_timestep)

                rewards += new_timestep.reward

                if new_timestep.last():
                    break

                timestep = new_timestep

                if agent.model_based_train:
                    agent.planning_update()

                agent.total_steps += 1

            rewards_window.append(rewards)
            t_window.append(t)

            if agent.episode % log_period == 0:
                agent.save_model()

                tf.summary.scalar("train/rewards", np.mean(rewards_window), step=episode)
                tf.summary.scalar("train/num_steps", np.mean(t_window), step=episode)
                agent.writer.flush()
                rewards_window.clear()
                t_window.clear()

                test(agent, environment, agent.episode, max_len=max_len, num_episodes=num_test_episodes)

            agent.episode += 1


def test(agent: Agent,
        environment: dm_env.Environment,
        episode: int,
        num_episodes: int,
        max_len: int,
        verbose: bool = False) -> None:

    total_reward = 0
    total_steps = 0

    for _ in range(num_episodes):
        timestep = environment.reset()

        for t in range(max_len):
            action = agent.policy(timestep, eval=True)
            new_timestep = environment.step(action)
            total_reward += new_timestep.reward

            if new_timestep.last():
                break

            timestep = new_timestep
            total_steps += 1

    total_reward /= num_episodes
    total_steps /= num_episodes

    tf.summary.scalar("test/num_steps", total_steps, step=episode)
    tf.summary.scalar("test/reward", total_reward, step=episode)
    agent.writer.flush()