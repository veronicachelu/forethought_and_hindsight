import dm_env
import tensorflow as tf
from prediction_agents import Agent
from utils.visualizer import *

def run_episodic(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        max_len: int,
        mdp_solver,
        model_class):

    agent.load_model()

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

        msve = get_rmsve(mdp_solver, hat_v)
        total_rmsve += msve
        agent.episode += 1
        avg_steps.append(t)

    return round(total_rmsve, 2), np.mean(avg_steps, dtype=int)

def get_rmsve(mdp_solver, hat_v):
    eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
    v = mdp_solver.get_optimal_v()
    rmsve = np.sqrt(np.sum(eta_pi * (v - hat_v) ** 2))
    return rmsve


def run_chain(agent: Agent,
        environment: dm_env.Environment,
        model_class,
        num_episodes: int,
        ):

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
        total_rmsve += np.sqrt(np.sum(np.power(hat_v - environment._true_v, 2)) / environment._nS)
        agent.episode += 1

        avg_steps.append(timesteps)

    return round(total_rmsve, 2), np.mean(avg_steps, dtype=int)