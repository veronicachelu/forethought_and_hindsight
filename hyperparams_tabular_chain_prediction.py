import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from absl import app
from absl import flags
from jax import random as jrandom
import network
import prediction_network
import itertools
from utils import *
import prediction_experiment
import agents
import prediction_agents
import utils
from agents import Agent

flags.DEFINE_string('run_mode', 'vanilla', 'what agent to run')
flags.DEFINE_string('policy', 'optimal', 'optimal or random')
# flags.DEFINE_string('model_class', 'linear', 'tabular or linear')
flags.DEFINE_string('model_class', 'tabular', 'tabular or linear')
# flags.DEFINE_string('env_type', 'continuous', 'discrete or continuous')
flags.DEFINE_string('env_type', 'discrete', 'discrete or continuous')
# flags.DEFINE_string('obs_type', 'onehot', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('obs_type', 'tile', 'onehot, tabular, tile for continuous')
flags.DEFINE_string('obs_type', 'tabular', 'onehot, tabular, tile for continuous')
flags.DEFINE_integer('max_reward', 1, 'max reward')
# flags.DEFINE_string('mdp', './continuous_mdps/obstacle.mdp',
# flags.DEFINE_string('mdp', 'boyan_chain', '')
flags.DEFINE_string('mdp', 'random_chain', '')
flags.DEFINE_integer('n_hidden_states', 14, 'num_states')
flags.DEFINE_integer('nS', 19, 'num_States')
flags.DEFINE_integer('env_size', 1, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
# flags.DEFINE_integer('env_size', 5, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
                                    'Continuous - Num of bins for each dimension of the discretization')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to run for.')
flags.DEFINE_integer('num_steps', 1000, 'Number of episodes to run for.')
flags.DEFINE_integer('runs', 100, 'Number of runs for each episode.')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', -1, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('num_units', 0, 'number of units per hidden layer')
flags.DEFINE_integer('planning_iter', 10, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('discount', 0.99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 1000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 100, 'min replay size before training.')
# flags.DEFINE_float('lr_model', 1, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 1e-2, 'learning rate for model optimizer')
flags.DEFINE_float('epsilon', 0.1, 'fraction of exploratory random actions at the end of the decay')
# flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions at the end of the decay')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_boolean('stochastic', False, 'stochastic transition dynamics or not.')
# flags.DEFINE_boolean('stochastic', True, 'stochastic transition dynamics or not.')
flags.DEFINE_boolean('random_restarts', False, 'random_restarts or not.')

FLAGS = flags.FLAGS


def run_episodic(agent: Agent,
        environment: dm_env.Environment,
        num_episodes: int,
        true_v):
    cumulative_reward = 0
    for episode in range(0, num_episodes):
        rewards = 0
        timestep = environment.reset()
        while True:
            # action = agent.policy(timestep)
            if FLAGS.mdp == "random_chain":
                action = agent._nrng.choice([0, 1], p=agent._pi[timestep.observation])
            elif FLAGS.mdp == "boyan_chain":
                action = 0
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

        cumulative_reward += rewards
        agent.episode += 1

    hat_v = agent._v_network
    rmsve = np.sqrt(np.sum(np.power(hat_v - true_v, 2)) / environment._nS)

    return rmsve

def run_experiment(run_mode, run, step, alpha, alpha_model, logs):
    nrng = np.random.RandomState(run)
    if FLAGS.mdp == "random_chain":
        env = RandomChain(rng=nrng,
                          nS=FLAGS.nS,
                          obs_type=FLAGS.obs_type
                          )
    elif FLAGS.mdp == "boyan_chain":
        env = BoyanChain(rng=nrng,
                          nS=FLAGS.n_hidden_states,
                          nF=FLAGS.nS,
                          obs_type=FLAGS.obs_type
                          )
    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape
    nS = env._nS
    policy = np.full((nS, nA), 1 / nA)

    rng = jrandom.PRNGKey(seed=run)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)

    v_network, v_network_params = prediction_network.get_prediction_v_network(
        num_hidden_layers=FLAGS.num_hidden_layers,
        num_units=FLAGS.num_units,
        nA=nA,
        input_dim=input_dim,
        rng=rng_q,
        model_class=FLAGS.model_class)
    model_network, model_network_params = prediction_network.get_prediction_model_network(
        num_hidden_layers=FLAGS.num_hidden_layers,
        num_units=FLAGS.num_units,
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        model_class=FLAGS.model_class)
    run_mode_to_agent_prop = {
        "vanilla": {"linear":
                        {"class": "VanillaLinearPrediction"},
                    "tabular":
                        {"class": "VanillaTabularPrediction"},
                    },
        "nstep_v1": {"linear":
                          {"class": "nStepLinearPredictionV1"},
                      "tabular":
                          {"class": "nStepTabularPredictionV1"},
                      },
        "nstep_v2": {"linear":
                         {"class": "nStepLinearPredictionV2"},
                     "tabular":
                         {"class": "nStepTabularPredictionV2"},
                     },
    }
    agent_prop = run_mode_to_agent_prop[FLAGS.run_mode]
    run_mode = FLAGS.run_mode
    agent_class = getattr(prediction_agents, agent_prop[FLAGS.model_class]["class"])

    agent = agent_class(run_mode=run_mode,
                       policy=policy,
                       action_spec=env.action_spec(),
                       v_network=v_network,
                       v_parameters=v_network_params,
                       model_network=model_network,
                       model_parameters=model_network_params,
                       batch_size=FLAGS.batch_size,
                       discount=FLAGS.discount,
                       replay_capacity=FLAGS.replay_capacity,
                       min_replay_size=FLAGS.min_replay_size,
                       model_learning_period=FLAGS.model_learning_period,
                       planning_iter=FLAGS.planning_iter,
                       planning_period=FLAGS.planning_period,
                       planning_depth=step,
                       lr=alpha,
                       lr_model=alpha_model,
                       epsilon=FLAGS.epsilon,
                       exploration_decay_period=FLAGS.num_episodes,
                       seed=FLAGS.seed,
                       rng=rng_agent,
                       nrng=nrng,
                       logs=logs,
                       max_len=FLAGS.max_len,
                       log_period=FLAGS.log_period,
                       input_dim=input_dim,
                       )

    rmsve = run_episodic(agent,
                           env,
                           FLAGS.num_episodes,
                           env._true_v,
                           )
    return rmsve

def main(argv):
    fig = plt.figure(figsize=(8, 4))
    del argv  # Unused.
    logs = os.path.join(os.path.join(FLAGS.logs, FLAGS.model_class), "chain")

    if not os.path.exists(logs):
        os.makedirs(logs)


    # all possible steps
    if FLAGS.run_mode == "vanilla":
        steps = [0]
        alphas = np.arange(0, 1.1, 0.1)
        alphas_model = [0.1]
    else:
        steps = np.power(2, np.arange(0, 8))
        alphas = [0.1]
        # alphas_model = np.arange(0.05, 0.15, 0.01)
        alphas_model = np.arange(0.05, 0.15, 0.01)

    checkpoint = os.path.join(logs, "hyperparams_rmsve_{}_{}.npy".format(FLAGS.mdp, FLAGS.run_mode))
    if os.path.exists(checkpoint):
        rmsve = np.load(checkpoint)
    else:
        # track the errors for each (step, alpha) combination
        rmsve = np.zeros((len(steps), len(alphas), len(alphas_model)))
        for run in tqdm(range(0, FLAGS.runs)):
            for step_ind, step in enumerate(steps):
                for alpha_ind, alpha in enumerate(alphas):
                    for alpha_ind_model, alpha_model in enumerate(alphas_model):
                        rmsve[step_ind, alpha_ind, alpha_ind_model] += run_experiment(FLAGS.run_mode,
                                                                     run,
                                                                     step,
                                                                     alpha,
                                                                     alpha_model,
                                                                     logs)
        # take average
        rmsve /= FLAGS.num_episodes * FLAGS.runs
        checkpoint = os.path.join(logs, "hyperparams_rmsve_{}_{}.npy".format(FLAGS.mdp, FLAGS.run_mode))
        np.save(checkpoint, rmsve)

    for i in range(0, len(steps)):
        if FLAGS.run_mode == "vanilla":
            plt.plot(alphas, rmsve[i, :], label='vanilla')
            plt.xlabel('alpha')
        else:
            ticks = np.arange(len(list(itertools.product(alphas, alphas_model))))
            ticks_labels = [str(a) for a in itertools.product(alphas, alphas_model)]
            plt.plot(ticks, np.reshape(rmsve[i, :], (-1)), label='n = %d' % (steps[i]))
            plt.xlabel('alpha/alpha_model')
            plt.xticks(ticks, ticks_labels)
    plt.ylabel('RMS error')
    # plt.ylim([0.064, 0.082])
    # plt.ylim([0.025, 0.06])
    plt.legend()

    plt.savefig(os.path.join(logs, 'hyperparams_{}_{}.png'.format(FLAGS.mdp, FLAGS.run_mode)))
    plt.close()

if __name__ == '__main__':
    app.run(main)

