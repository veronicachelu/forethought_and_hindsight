import dm_env
import tensorflow as tf

from agents.agent import Agent
from utils.visualizer import *
from utils.mdp_solvers.solve_mdp import MdpSolver

def run_episodic(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        max_len: int,
        space,
        aux_agent_configs
        ):
    if space["agent_config"]["model_family"] == "q_true" and \
                    space["env_config"]["model_class"] == "tabular":
        mdp_solver = MdpSolver(environment, 48, space["env_config"]["nA"], aux_agent_configs["discount"])
        agent._o_network, agent._fw_o_network, r, agent._true_discount = mdp_solver.get_true_action_model()
        agent._r_network = np.sum(r, axis=0)
    # else:
    #     mdp_solver = MdpSolver(environment, 48, space["env_config"]["nA"], aux_agent_configs["discount"])
    #
    #     _, _, _, agent._true_discount = mdp_solver.get_true_action_model()
    # _, _, agent._true_r_network, agent._true_discount = mdp_solver.get_true_action_model()
    cumulative_reward = 0
    # agent.load_model()
    ep_steps = []
    ep_rewards = []
    all_states = environment.get_all_states()

    with agent.writer.as_default():
        for episode in np.arange(start=agent.episode, stop=num_episodes):
            # Run an episode.
            ep_reward = 0
            timestep = environment.reset()
            agent.update_hyper_params(agent.episode, num_episodes)
            for t in range(max_len):
                action = agent.policy(timestep)
                new_timestep = environment.step(action)

                agent.save_transition(timestep, action, new_timestep)

                if agent.model_based_train():
                    agent.model_update(timestep, action, new_timestep)

                if agent.model_free_train() and not aux_agent_configs["mb"]:
                    agent.value_update(timestep, action, new_timestep)

                ep_reward += new_timestep.reward

                if agent.model_based_train():
                    if aux_agent_configs["mb"] and \
                        aux_agent_configs["pivot"] == "c" and\
                        timestep.discount is None and\
                       aux_agent_configs["agent_type"] == "fw":
                        agent.planning_update(timestep)

                    if aux_agent_configs["pivot"] == "c":
                        agent.planning_update(new_timestep)
                    else:
                        agent.planning_update(timestep)

                tf.summary.scalar("train/ep_steps", t, step=agent.total_steps)
                tf.summary.scalar("train/cum_reward", cumulative_reward, step=agent.total_steps)
                tf.summary.scalar("train/step_reward", new_timestep.reward, step=agent.total_steps)

                if new_timestep.last():
                    if aux_agent_configs["mb"] and aux_agent_configs["pivot"] == "p" and\
                        aux_agent_configs["agent_type"] == "bw":
                        agent.planning_update(new_timestep)

                    break

                timestep = new_timestep

                agent.total_steps += 1

            # hat_p = agent.get_model_for_all_states(all_states)
            # plot_p(env=environment,
            #        p=hat_p,
            #        logs=agent._images_dir,
            #        episode=agent.episode,
            #        timestep=t)

            # hat_q = agent.get_qvalues_for_all_states(all_states)
            # # hat_pi = agent.get_policy_for_all_states(environment.get_all_states())
            # # _hat_pi_ = environment.reshape_pi(hat_pi)
            # # _hat_v_ = environment.reshape_v(hat_v)
            # _hat_q_ = environment.reshape_q(hat_q)
            # plot_q(env=environment,
            #        action_values=_hat_q_,
            #        logs=agent._images_dir,
            #        filename="a_{}.png".format(agent.episode))
            # plot_pi_from_q(env=environment,
            #                q=_hat_q_,
            #                logs=agent._images_dir,
            #                filename="pi_{}.png".format(agent.episode))

            cumulative_reward += ep_reward
            agent.episode += 1



            ep_steps.append(t)
            ep_rewards.append(ep_reward)


            tf.summary.scalar("train/reward", np.mean(ep_reward), step=agent.episode)
            # tf.summary.scalar("train/avg_reward", np.mean(ep_rewards), step=agent.episode)
            # tf.summary.scalar("train/avg_steps", np.mean(ep_steps), step=agent.episode)
            tf.summary.scalar("train/steps", t, step=agent.episode)
            agent.writer.flush()

            # if agent.episode % 10 == 0:
            #     test_agent(agent, environment, 10, 1000)

    # agent.save_model()

    # avg_steps = np.mean(ep_steps) if len(ep_steps) > 0 else None
    # avg_reward = np.mean(ep_rewards) if len(ep_rewards) > 0 else None

    return cumulative_reward, agent.total_steps

def test_agent(agent, environment, num_episodes, max_len):
    cumulative_reward = 0
    ep_steps = []
    ep_rewards = []
    for _ in np.arange(start=0, stop=num_episodes):
        # Run an episode.
        rewards = 0
        timestep = environment.reset()
        for t in range(max_len):
            action = agent.policy(timestep, eval=True)
            new_timestep = environment.step(action)

            rewards += new_timestep.reward
            if new_timestep.last():
                break

            timestep = new_timestep

        cumulative_reward += rewards

        ep_steps.append(t)
        ep_rewards.append(rewards)

    tf.summary.scalar("test/avg_reward", np.mean(ep_rewards), step=agent.episode)
    tf.summary.scalar("test/avg_steps", np.mean(ep_steps), step=agent.episode)
    agent.writer.flush()

