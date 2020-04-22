import dm_env
import tensorflow as tf
from agents import Agent
from utils.visualizer import *
import contextlib

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

def run_episodic(agent: Agent,
                 environment: dm_env.Environment,
                 mdp_solver,
                 space,
                 aux_agent_configs,
               ):
    # ls = "learning"
    # ls = "models"
    ls = "planning_full"
    # ls = "planning"

    weighted = False if space["env_config"]["non_gridworld"] else True
    with agent.writer.as_default() if space["plot_curves"] and agent._logs is not None else dummy_context_mgr():
        agent.load_model()
        if ls == "models":
            agent.load_v()
        elif ls == "planning" or ls == "planning_full":
            agent.load_m()

        total_rmsve = 0
        total_reward = 0
        ep_rewards = []
        ep_rmsves = []
        ep_steps = []
        for episode in np.arange(start=agent.episode, stop=space["env_config"]["num_episodes"]):
            # Run an episode.
            rewards = 0
            ep_rmsve = 0
            t = 0
            timestep = environment.reset()
            agent.update_hyper_params(episode, space["env_config"]["num_episodes"])
            while True:
                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                if ls == "models":
                    if agent.model_based_train():
                        agent.save_transition(timestep, action, new_timestep)
                        agent.model_update(timestep, action, new_timestep)

                if ls == "learning":
                    if agent.model_free_train():
                        agent.value_update(timestep, action, new_timestep)

                rewards += new_timestep.reward

                if ls == "planning_full":
                    if agent.model_free_train():
                        agent.planning_value_update(timestep, action, new_timestep)

                    if agent.model_based_train():
                        agent.planning_update(timestep)

                if ls == "planning":
                    agent.planning_update(new_timestep)

                agent.total_steps += 1
                t += 1

                if new_timestep.last() or (aux_agent_configs["max_len"] is not None and \
                                                   t == aux_agent_configs["max_len"]):
                    break

                timestep = new_timestep
            if space["env_config"]["env_type"] != "continuous":
                hat_v = agent._v_network if space["env_config"]["model_class"] == "tabular" \
                    else agent.get_values_for_all_states(environment.get_all_states(), ls)
                hat_error = np.abs(environment._true_v - hat_v)
                rmsve = get_rmsve(environment, mdp_solver, hat_v, environment._true_v, weighted=weighted)
                ep_rmsve += rmsve
            else:
                hat_v = agent.get_value_for_state(timestep.observation)
                v = mdp_solver.get_value_for_state(timestep.observation)
                hat_error = np.abs(v - hat_v)
                rmsve = np.power(v - hat_v, 2)
                ep_rmsve += rmsve

            total_rmsve += rmsve
            total_reward += rewards
            ep_steps.append(t)
            ep_rmsves.append(rmsve)
            ep_rewards.append(rewards)

            if space["plot_errors"] and agent.episode % space["log_period"] == 0 and\
                    not space["env_config"]["non_gridworld"] and \
                    space["env_config"]["env_type"] != "continuous":
                plot_error(env=environment,
                           values=(environment.reshape_v(mdp_solver.get_optimal_v()) - environment.reshape_v(
                               hat_v)) ** 2,
                           logs=agent._images_dir,
                           eta_pi=environment.reshape_v(mdp_solver.get_eta_pi(mdp_solver._pi)),
                           filename="error_{}.png".format(agent.episode))
            if space["plot_values"] and agent.episode % space["log_period"] == 0 and\
                    not space["env_config"]["non_gridworld"] and \
                    space["env_config"]["env_type"] != "continuous":
                plot_v(env=environment,
                       values=environment.reshape_v(hat_v),
                       logs=agent._images_dir,
                       true_v=environment.reshape_v(mdp_solver.get_optimal_v()),
                       filename="v_{}.png".format(agent.episode))

            if space["plot_curves"] and agent.episode % space["log_period"] == 0:
                # agent.save_model()
                tf.summary.scalar("train/rmsve", rmsve, step=agent.episode)
                tf.summary.scalar("train/rewards", rewards, step=agent.episode)
                tf.summary.scalar("train/steps", t, step=agent.episode)
                tf.summary.scalar("train/total_rmsve", total_rmsve, step=agent.total_steps)
                tf.summary.scalar("train/total_reward", total_reward, step=agent.total_steps)
                tf.summary.scalar("train/avg_rmsve", np.mean(ep_rmsves), step=agent.episode)
                tf.summary.scalar("train/avg_reward", np.mean(ep_rewards), step=agent.episode)
                tf.summary.scalar("train/avg_steps", np.mean(ep_steps, dtype=int), step=agent.episode)
                agent.writer.flush()

            agent.episode += 1

        if ls == "learning":
            agent.save_v()
        elif ls == "models":
            agent.save_m()
        # agent.save_v()
        # agent.save_m()
        rmsve_start = 0
        # rmsve_start = np.power(environment._true_v - hat_v, 2)[environment._get_state_index(environment._sX, environment._sY)]
        return round(total_rmsve, 2), round(rmsve, 2), round(rmsve_start, 2), \
               np.mean(ep_steps, dtype=int), hat_v, hat_error



def get_rmsve(env, mdp_solver, hat_v, v, weighted=False):
    if weighted:
        eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
        rmsve = np.sqrt(np.sum(eta_pi * (v - hat_v) ** 2))
    else:
        rmsve = np.sqrt(np.sum(np.power(v - hat_v, 2)) / env._nS)
    return rmsve
