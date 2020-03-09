from agents import Agent
import dm_env
import tensorflow as tf
import numpy as np
from utils.visualizer import *
from utils.solve_mdp import MdpSolver

def run_episodic(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        num_test_episodes: int,
        max_len: int,
        log_period: 10,
        mdp_solver,
        model_class,
        verbose: bool = False) -> None:

    agent.load_model()

    cumulative_reward = 0
    with agent.writer.as_default():
        for episode in np.arange(start=agent.episode, stop=num_episodes):
            # Run an episode.
            rewards = 0
            timestep = environment.reset()
            agent.update_hyper_params(episode, num_episodes)
            for t in range(max_len):

                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                if agent.model_based_train():
                    agent.save_transition(timestep, action, new_timestep)
                    agent.model_update(timestep, action, new_timestep)

                if agent.model_free_train():
                    agent.value_update(timestep, action, new_timestep)

                rewards += new_timestep.reward

                if agent.model_based_train:
                    agent.planning_update(timestep)

                if new_timestep.last():
                    break

                timestep = new_timestep

                agent.total_steps += 1
                # if agent.total_steps % log_period == 0:
                #     hat_v = agent._v_network if model_class == "tabular" \
                #         else agent.get_values_for_all_states(environment.get_all_states())
                #     plot_v(env=environment,
                #            values=environment.reshape_v(hat_v),
                #            logs=agent._images_dir,
                #            true_v=mdp_solver.get_optimal_v(),
                #            filename="v_{}_{}.png".format(agent.episode, agent.total_steps))

            cumulative_reward += rewards
            if agent.episode % log_period == 0:
                agent.save_model()

                hat_v = agent._v_network if model_class == "tabular" \
                    else agent.get_values_for_all_states(environment.get_all_states())
                plot_v(env=environment,
                       values=environment.reshape_v(hat_v),
                       logs=agent._images_dir,
                       true_v=mdp_solver.get_optimal_v(),
                       filename="v_{}.png".format(agent.episode))
                # plot_v(env=environment,
                #        values=environment.reshape_v(agent._v_network),
                #        logs=agent._images_dir,
                #        true_v=mdp_solver.get_optimal_v(),
                #        filename="v_{}.png".format(agent.episode))

                tf.summary.scalar("train/msve", get_msve(mdp_solver, agent._pi,
                                                         hat_v),
                                  step=agent.episode)

                tf.summary.scalar("train/rewards", np.mean(rewards), step=episode)
                tf.summary.scalar("train/num_steps", np.mean(t), step=episode)
                tf.summary.scalar("train/cumulative_reward", cumulative_reward, step=episode)
                agent.writer.flush()

                # test(agent, environment, agent.episode, max_len=max_len, num_episodes=num_test_episodes)

            agent.episode += 1

def run(agent: Agent,
        environment: dm_env.Environment,
        num_steps: int,
        log_period: 10,
        mdp_solver,
        model_class,
        verbose: bool = False) -> None:

    agent.load_model()

    cumulative_reward = 0
    timestep = environment.reset()
    with agent.writer.as_default():
        for step in np.arange(start=agent.total_steps, stop=num_steps):
            agent.update_hyper_params(step, num_steps)
            action = agent.policy(timestep)
            new_timestep = environment.step(action)

            if agent.model_based_train():
                agent.save_transition(timestep, action, new_timestep)
                agent.model_update(timestep, action, new_timestep)

            if agent.model_free_train():
                agent.value_update(timestep, action, new_timestep)

            cumulative_reward += new_timestep.reward

            if agent.model_based_train:
                agent.planning_update(timestep)

            if agent.total_steps % log_period == 0:
                agent.save_model()
                print("Plotting v for episode {}, total_steps {}".format(agent.episode,

                                                                            agent.total_steps))
                hat_v = agent._v_network if model_class == "tabular" \
                    else agent.get_values_for_all_states(environment.get_all_states())
                plot_v(env=environment,
                       values=environment.reshape_v(hat_v),
                       logs=agent._images_dir,
                       true_v=mdp_solver.get_optimal_v(),
                       filename="v_{}.png".format(agent.total_steps))

                tf.summary.scalar("train/cumulative_reward", cumulative_reward, step=agent.total_steps)
                tf.summary.scalar("train/msve", get_msve(mdp_solver, agent._pi, hat_v), step=agent.total_steps)
                agent.writer.flush()

            if new_timestep.last():
                timestep = environment.reset()
            else:
                timestep = new_timestep

            agent.total_steps += 1

def get_msve(mdp_solver, pi, hat_v):
    eta_pi = mdp_solver.get_eta_pi(pi)
    v = mdp_solver.get_optimal_v()
    # msve = np.sum([eta_pi[s] * ((v[s] - hat_v[s]) ** 2) for s in range(mdp_solver._nS)])
    msve = np.sum(eta_pi * (v - hat_v) ** 2)
    return msve

def test(agent: Agent,
        environment: dm_env.Environment,
        episode: int,
        num_episodes: int,
        max_len: int,
        verbose: bool = False) -> None:

    total_reward = 0
    total_steps = 0
    cumulative_steps = 0
    cumulative_reward = 0

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
            cumulative_steps += 1

    cumulative_reward = total_reward
    total_reward /= num_episodes
    total_steps /= num_episodes

    tf.summary.scalar("test/num_steps", total_steps, step=episode)
    # tf.summary.scalar("test/cumulative_steps", cumulative_steps, step=episode)
    tf.summary.scalar("test/cumulative_reward", cumulative_reward, step=episode)
    tf.summary.scalar("test/reward", total_reward, step=episode)
    agent.writer.flush()