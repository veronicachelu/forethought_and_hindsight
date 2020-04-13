import dm_env
import tensorflow as tf
from agents import Agent
from utils.visualizer import *

def run_episodic(agent: Agent,
                 environment: dm_env.Environment,
                 num_episodes: int,
                 max_len: int,
                 mdp_solver,
                 model_class,
                 log_period=1,
                 plot_values=False,
                 plot_curves=False,
                 plot_errors=False,):

    if plot_errors:
        with agent.writer.as_default():
            return run_mdp_forall_episodes(agent=agent,
                                      environment=environment,
                                      num_episodes =num_episodes,
                                      max_len=max_len,
                                      model_class=model_class,
                                      log_period=log_period,
                                      mdp_solver=mdp_solver,
                                      plot_values=plot_values,
                                      plot_curves=plot_curves,
                                      plot_errors=plot_errors)
    else:
        return run_mdp_forall_episodes(agent=agent,
                                      environment=environment,
                                      num_episodes =num_episodes,
                                      max_len=max_len,
                                      model_class=model_class,
                                      log_period=log_period,
                                      mdp_solver=mdp_solver,
                                      plot_values=plot_values,
                                      plot_curves=plot_curves,
                                      plot_errors=plot_errors)

def run_mdp_forall_episodes(
        agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        max_len: int,
        mdp_solver,
        model_class,
        log_period=1,
        plot_values=False,
        plot_curves=False,
        plot_errors=False,):

    # agent.load_model()

    total_rmsve = 0
    avg_steps = []
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

            if agent.model_based_train():
                agent.planning_update(timestep)

            if new_timestep.last():
                break

            timestep = new_timestep
            agent.total_steps += 1

        hat_v = agent._v_network if model_class == "tabular" \
            else agent.get_values_for_all_states(environment.get_all_states())
        hat_error = np.abs(mdp_solver.get_optimal_v() - hat_v)

        rmsve = get_rmsve(mdp_solver, hat_v)
        total_rmsve += rmsve

        # if plot_errors and agent.episode % log_period == 0:
        #     plot_error(env=environment,
        #                values=(environment.reshape_v(mdp_solver.get_optimal_v()) - environment.reshape_v(hat_v)) ** 2,
        #                logs=agent._images_dir,
        #                eta_pi=environment.reshape_v(mdp_solver.get_eta_pi(mdp_solver._pi)),
        #                filename="error_{}.png".format(agent.episode))
        # if plot_values and agent.episode % log_period == 0:
        #     plot_v(env=environment,
        #            values=environment.reshape_v(hat_v),
        #            logs=agent._images_dir,
        #            true_v=environment.reshape_v(mdp_solver.get_optimal_v()),
        #            filename="v_{}.png".format(agent.episode))

        if plot_curves and agent.episode % log_period == 0:
            # agent.save_model()
            tf.summary.scalar("train/rmsve", rmsve, step=agent.episode)
            tf.summary.scalar("train/rewards", np.mean(rewards), step=agent.episode)
            tf.summary.scalar("train/steps", np.mean(t), step=agent.episode)
            tf.summary.scalar("train/total_rmsve", total_rmsve, step=agent.total_steps)
            tf.summary.scalar("train/avg_steps", np.mean(avg_steps, dtype=int), step=agent.total_steps)
            agent.writer.flush()

        agent.episode += 1
        avg_steps.append(t)

    return round(total_rmsve, 2), np.mean(avg_steps, dtype=int), hat_v, hat_error

def get_rmsve(mdp_solver, hat_v):
    eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
    v = mdp_solver.get_optimal_v()
    rmsve = np.sqrt(np.sum(eta_pi * (v - hat_v) ** 2))
    return rmsve

def run_chain(agent: Agent,
              environment: dm_env.Environment,
              num_episodes: int,
              model_class,
              mdp_solver,
              log_period=1,
              plot_values=False,
              plot_curves=False,
              plot_errors=False,):

    if plot_errors:
        with agent.writer.as_default():
            return run_chain_forall_episodes(agent=agent,
                                      environment=environment,
                                      num_episodes =num_episodes,
                                      model_class=model_class,
                                      mdp_solver=mdp_solver,
                                      log_period=log_period,
                                      plot_values=plot_values,
                                      plot_curves=plot_curves,
                                      plot_errors=plot_errors)
    else:
        return run_chain_forall_episodes(agent=agent,
                                      environment=environment,
                                      num_episodes =num_episodes,
                                      model_class=model_class,
                                      mdp_solver=mdp_solver,
                                      log_period=log_period,
                                      plot_values=plot_values,
                                      plot_curves=plot_curves,
                                      plot_errors=plot_errors)

def run_chain_forall_episodes(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        model_class,
        mdp_solver,
        log_period=1,
        plot_values=False,
        plot_curves=False,
        plot_errors=False):

    total_rmsve = 0
    pred_timestep = None
    avg_steps = []
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
        hat_error = np.abs(mdp_solver.get_optimal_v() - hat_v)
        rmsve = np.sqrt(np.sum(np.power(hat_v - environment._true_v, 2)) / environment._nS)
        total_rmsve += rmsve

        if plot_errors and agent.episode % log_period == 0:
            tf.summary.scalar("train/rmsve", rmsve, step=agent.episode)
            tf.summary.scalar("train/steps", timesteps, step=agent.episode)
            tf.summary.scalar("train/total_rmsve", total_rmsve, step=agent.total_steps)
            tf.summary.scalar("train/avg_steps", np.mean(avg_steps, dtype=int), step=agent.total_steps)
            agent.writer.flush()

        agent.episode += 1
        avg_steps.append(timesteps)

    return round(total_rmsve, 2), np.mean(avg_steps, dtype=int), hat_v, hat_error