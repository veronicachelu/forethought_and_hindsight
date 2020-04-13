import dm_env
import tensorflow as tf

from agents import Agent
from utils.visualizer import *
NON_GRIDWORLD_MDPS = ["random_chain", "boyan_chain", "bandit", "loopy_chain", "tree"]

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
    total_msve = 0
    with agent.writer.as_default():
        for episode in np.arange(start=agent.episode, stop=num_episodes):
            # Run an episode.
            rewards = 0
            timestep = environment.reset()
            # agent.update_hyper_params(episode, num_episodes)
            for t in range(max_len):

                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                if agent.model_based_train():
                    agent.save_transition(timestep, action, new_timestep)
                    agent.model_update(timestep, action, new_timestep)

                if agent.model_free_train():
                    agent.value_update(timestep, action, new_timestep)

                rewards += new_timestep.reward

                if agent.model_based_train():
                    # agent.planning_update(timestep)
                    agent.planning_update(new_timestep)

                if new_timestep.last():
                    break

                timestep = new_timestep

                agent.total_steps += 1

            cumulative_reward += rewards
            hat_v = agent._v_network if model_class == "tabular" \
                else agent.get_values_for_all_states(environment.get_all_states())

            msve = get_msve(mdp_solver, hat_v)
            if agent.episode % log_period == 0:
                agent.save_model()

                # plot_v(env=environment,
                #        values=environment.reshape_v(agent._v_network),
                #        logs=agent._images_dir,
                #        true_v=mdp_solver.get_optimal_v(),
                #        filename="v_{}.png".format(agent.episode))

                plot_error(env=environment,
                       values=(environment.reshape_v(mdp_solver.get_optimal_v()) - environment.reshape_v(hat_v)) ** 2,
                       logs=agent._images_dir,
                       eta_pi= environment.reshape_v(mdp_solver.get_eta_pi(mdp_solver._pi)),
                       filename="error_{}.png".format(agent.episode))
                plot_v(env=environment,
                       values=environment.reshape_v(hat_v),
                       logs=agent._images_dir,
                       true_v=environment.reshape_v(mdp_solver.get_optimal_v()),
                       filename="v_{}.png".format(agent.episode))

                total_msve += msve
                tf.summary.scalar("train/msve", msve,
                                  step=agent.episode)
                tf.summary.scalar("train/rewards", np.mean(rewards), step=agent.episode)
                tf.summary.scalar("train/num_steps", np.mean(t), step=agent.episode)
                # tf.summary.scalar("train/cumulative_reward", cumulative_reward, step=agent.episode)
                agent.writer.flush()

                # test(agent, environment, agent.episode, max_len=max_len, num_episodes=num_test_episodes)
            tf.summary.scalar("train/cumm_msve", total_msve,
                              step=agent.total_steps)
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
            # agent.update_hyper_params(step, num_steps)
            action = agent.policy(timestep)
            new_timestep = environment.step(action)

            if agent.model_based_train():
                agent.save_transition(timestep, action, new_timestep)
                agent.model_update(timestep, action, new_timestep)

            if agent.model_free_train():
                agent.value_update(timestep, action, new_timestep)

            cumulative_reward += new_timestep.reward

            if agent.model_based_train():
                agent.planning_update(timestep)

            if agent.total_steps % log_period == 0:
                agent.save_model()
                # print("Plotting v for episode {}, total_steps {}".format(agent.episode,
                #
                #                                                             agent.total_steps))
                hat_v = agent._v_network if model_class == "tabular" \
                    else agent.get_values_for_all_states(environment.get_all_states())
                plot_v(env=environment,
                       values=environment.reshape_v(hat_v),
                       logs=agent._images_dir,
                       true_v=mdp_solver.get_optimal_v(),
                       filename="v_{}.png".format(agent.total_steps))

                # tf.summary.scalar("train/cumulative_reward", cumulative_reward, step=agent.total_steps)
                tf.summary.scalar("train/msve", get_msve(mdp_solver, agent._pi, hat_v), step=agent.total_steps)
                agent.writer.flush()

            if new_timestep.last():
                timestep = environment.reset()
            else:
                timestep = new_timestep

            agent.total_steps += 1

def get_msve(mdp_solver, hat_v):
    eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
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
    # tf.summary.scalar("test/cumulative_reward", cumulative_reward, step=episode)
    tf.summary.scalar("test/reward", total_reward, step=episode)
    agent.writer.flush()


def run_chain(agent: Agent,
        environment: dm_env.Environment,
        mdp,
        model_class,
        seed: int,
        # num_runs: int,
        num_episodes: int,
        log_period: 1,
        # total_rmsve
              ):

    cumulative_reward = 0
    cumulative_rmsve = 0
    pred_timestep = None
    rmsve = np.zeros((num_episodes//log_period))
    with agent.writer.as_default():
        for episode in np.arange(start=agent.episode, stop=num_episodes):
            rewards = 0
            timestep = environment.reset()
            timesteps = 0
            agent.update_hyper_params(episode, num_episodes)
            while True:
                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                if agent.model_based_train():
                    agent.save_transition(timestep, action, new_timestep)
                    agent.model_update(timestep, action, new_timestep)

                if agent.model_free_train():
                    agent.value_update(timestep, action, new_timestep)

                rewards += new_timestep.reward

                if agent.model_based_train():
                    agent.planning_update(timestep, pred_timestep)

                if new_timestep.last():
                    break

                pred_timestep = timestep
                timestep = new_timestep
                agent.total_steps += 1
                timesteps += 1

            hat_v = agent._v_network if model_class == "tabular" \
                else agent.get_values_for_all_states(environment.get_all_states())
            cumulative_rmsve += np.sqrt(np.sum(np.power(hat_v - environment._true_v, 2)) / environment._nS)
            if agent.episode % log_period == 0:
                rmsve[episode // log_period] = np.sqrt(
                    np.sum(np.power(hat_v - environment._true_v, 2)) / environment._nS)
                # total_rmsve[episode // log_period] += (1 / num_runs) * rmsve[episode // log_period]
                tf.summary.scalar("train/rmsve", rmsve[episode // log_period], step=agent.episode)
                tf.summary.scalar("train/cumulative_rmsve", cumulative_rmsve, step=agent.episode)
                tf.summary.scalar("train/steps", timesteps, step=agent.episode)
                # tf.summary.scalar("train/total_rmsve_{}".format(seed), total_rmsve[episode // log_period], step=agent.episode)
                agent.writer.flush()

            cumulative_reward += rewards
            agent.episode += 1

    # return total_rmsve