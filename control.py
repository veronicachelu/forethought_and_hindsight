from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm

import control_agents
import control_experiment
import utils
from utils import *

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
# flags.DEFINE_string('mdp', './mdps/simple.mdp',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
# flags.DEFINE_string('mdp', 'boyan_chain',
# flags.DEFINE_string('mdp', 'random_chain',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_48.mdp',
flags.DEFINE_string('mdp', './mdps/maze_80.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_221.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_486.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_864.mdp',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
# flags.DEFINE_integer('env_size', 20, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
flags.DEFINE_integer('env_size', 1, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
# flags.DEFINE_integer('env_size', 5, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
                                    'Continuous - Num of bins for each dimension of the discretization')
# flags.DEFINE_integer('n_hidden_states', 5, 'max reward')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
# flags.DEFINE_integer('num_episodes', 1800, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 180, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 70, 'Number of episodes to run for.')
flags.DEFINE_integer('num_episodes', 50, 'Number of episodes to run for.')
flags.DEFINE_integer('num_runs', 1, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_runs', 0, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_steps', 2000, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_steps', 1000, 'Number of episodes to run for.')
flags.DEFINE_integer('num_steps', 500, 'Number of episodes to run for.')
flags.DEFINE_integer('num_test_episodes', 100, 'Number of test episodes to run for.')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
# flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('num_units', 0, 'number of units per hidden layer')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('planning_depth', 0, 'Planning depth for MCTS')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
# flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 1000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 100, 'min replay size before training.')
flags.DEFINE_float('lr', 5e-1, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 5e-3, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 1, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 0.2, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 0.2, 'learning rate for q optimizer')
flags.DEFINE_float('lr_planning', 5e-1, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 1e-3, 'learning rate for q optimizer')
# flags.DEFINE_float('lr_model', 1e-2, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 0.01, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 1e-3, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 1e-3, 'learning rate for model optimizer')
flags.DEFINE_float('lr_model',  1e-1, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 0.1, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 5e-4, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 1e-3, 'learning rate for model optimizer')
flags.DEFINE_float('epsilon', 0.1, 'fraction of exploratory random actions at the end of the decay')
# flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions at the end of the decay')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_boolean('stochastic', False, 'stochastic transition dynamics or not.')
# flags.DEFINE_boolean('stochastic', True, 'stochastic transition dynamics or not.')
flags.DEFINE_boolean('random_restarts', False, 'random_restarts or not.')

FLAGS = flags.FLAGS

run_mode_to_agent_prop = {
        "vanilla": {"linear":
                        {"class": "VanillaLinearControl"},
                    "tabular":
                        {"class": "VanillaTabularControl"},
                    },
        "pred_exp": {"linear":
                       {"class": "nStepLcPredExp"},
                   "tabular":
                       {"class": "nStepTcPredDistrib"},
                   },
        "pred_gen": {"linear":
                       {"class": "nStepLcPredGen"},
                   "tabular":
                       {"class": "nStepTcPredGen"},
                   },
        "jumpy_exp": {"linear":
                      {"class": "nStepLcJumpyExp"},
                  "tabular":
                      {"class": "nStepTcJumpyDistrib"},
                  },
        "jumpy_gen": {"linear":
                      {"class": "nStepLcJumpyGen"},
                  "tabular":
                      {"class": "nStepTcJumpyGen"},
                  },
        "jumpy_fwbw_gen": {"linear":
                      {"class": "nStepLcJumpyFwBwGen"},
                  "tabular":
                      {"class": "nStepTcJumpyFwBwGen"},
                  },
    }

def get_env(nrng, logs):
    if FLAGS.mdp == "random_chain":
        env = RandomChain(rng=nrng,
                          nS=FLAGS.env_size,
                          obs_type=FLAGS.obs_type
                          )
        nS = env._nS
    elif FLAGS.mdp == "boyan_chain":
        env = BoyanChain(rng=nrng,
                          nS=FLAGS.n_hidden_states,
                          nF=FLAGS.env_size,
                          obs_type=FLAGS.obs_type
                          )
        nS = env._nF
    else:
        envs = {"discrete": {"class": "MicroWorld"},
                "continuous": {"class": "ContinuousWorld"}
                }
        env_class = getattr(utils, envs[FLAGS.env_type]["class"])
        env = env_class(path=FLAGS.mdp,
                        stochastic=FLAGS.stochastic,
                        random_restarts=FLAGS.random_restarts,
                        seed=FLAGS.seed,
                        rng=nrng,
                        obs_type=FLAGS.obs_type,
                        env_size=FLAGS.env_size)
        nS = env._nS

    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape

    if FLAGS.mdp == "random_chain" or FLAGS.mdp == "boyan_chain":
        policy = np.full((nS, nA), 1 / nA)
        mdp_solver = None
    else:
        plot_grid(env, logs, env_type=FLAGS.env_type)
        mdp_solver = MdpSolver(env, nS, nA, FLAGS.discount)
        policy = mdp_solver.get_optimal_policy()
        v = mdp_solver.get_optimal_v()
        v = env.reshape_v(v)
        plot_v(env, v, logs, env_type=FLAGS.env_type)
        plot_policy(env, env.reshape_pi(policy), logs, env_type=FLAGS.env_type)
        eta_pi = mdp_solver.get_eta_pi(policy)
        plot_eta_pi(env, env.reshape_v(eta_pi), logs, env_type=FLAGS.env_type)

    return env, nS, nA, input_dim, policy, mdp_solver

def get_agent(env, seed, nrng, nA, input_dim, policy, logs):
    rng = jrandom.PRNGKey(seed=seed)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    q_network, q_network_params = prediction_network.get_control_v_network(num_hidden_layers=FLAGS.num_hidden_layers,
                                                                              num_units=FLAGS.num_units,
                                                                              nA=nA,
                                                                              input_dim=input_dim,
                                                                              rng=rng_q,
                                                                              model_class=FLAGS.model_class)
    model_network, model_network_params = prediction_network.get_control_model_network(
        num_hidden_layers=FLAGS.num_hidden_layers,
        num_units=FLAGS.num_units,
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        model_class=FLAGS.model_class)

    agent_prop = run_mode_to_agent_prop[FLAGS.run_mode]
    run_mode = FLAGS.run_mode
    agent_class = getattr(control_agents, agent_prop[FLAGS.model_class]["class"])

    agent = agent_class(
        run_mode=run_mode,
        policy=policy,
        action_spec=env.action_spec(),
        q_network=q_network,
        q_parameters=q_network_params,
        model_network=model_network,
        model_parameters=model_network_params,
        batch_size=FLAGS.batch_size,
        discount=FLAGS.discount,
        replay_capacity=FLAGS.replay_capacity,
        min_replay_size=FLAGS.min_replay_size,
        model_learning_period=FLAGS.model_learning_period,
        planning_iter=FLAGS.planning_iter,
        planning_period=FLAGS.planning_period,
        planning_depth=FLAGS.planning_depth,
        lr=FLAGS.lr,
        lr_model=FLAGS.lr_model,
        lr_planning=FLAGS.lr_planning,
        epsilon=FLAGS.epsilon,
        exploration_decay_period=FLAGS.num_episodes,
        seed=seed,
        rng=rng_agent,
        nrng=nrng,
        logs=logs,
        max_len=FLAGS.max_len,
        log_period=FLAGS.log_period,
        input_dim=input_dim,
    )
    return agent

def run_experiment(seed, logs):
    nrng = np.random.RandomState(seed)
    env, nS, nA, input_dim, policy, mdp_solver = get_env(nrng, logs)
    agent = get_agent(env, seed, nrng, nA, input_dim, policy, logs)
    return env, agent, mdp_solver

def main(argv):
    del argv  # Unused.
    mdp_filename = os.path.splitext(os.path.basename(FLAGS.mdp))[0]
    logs = os.path.join(FLAGS.logs, FLAGS.model_class)
    logs = os.path.join(logs, mdp_filename)
    if FLAGS.max_len != -1:
        logs = os.path.join(logs, "episodic")
    else:
        logs = os.path.join(logs, "absorbing")

    logs = os.path.join(logs, "stochastic" if FLAGS.stochastic else "deterministic")

    if not os.path.exists(logs):
        os.makedirs(logs)

    if FLAGS.mdp == "random_chain" or FLAGS.mdp == "boyan_chain":
        for seed in tqdm(range(0, FLAGS.num_runs)):
            env, agent, _ = run_experiment(seed, logs)

            control_experiment.run_chain(
                agent=agent,
                mdp=FLAGS.mdp,
                model_class=FLAGS.model_class,
                seed=seed,
                environment=env,
                num_episodes=FLAGS.num_episodes,
                log_period=FLAGS.log_period,
            )
    else:
        for seed in tqdm(range(0, FLAGS.num_runs)):
            env, agent, mdp_solver = run_experiment(seed, logs)
            if FLAGS.max_len == -1:
                control_experiment.run(
                    agent=agent,
                    environment=env,
                    mdp_solver=mdp_solver,
                    model_class=FLAGS.model_class,
                    num_steps=FLAGS.num_steps,
                    log_period=FLAGS.log_period,
                    verbose=FLAGS.verbose
                )
            else:
                control_experiment.run_episodic(
                    agent=agent,
                    environment=env,
                    mdp_solver=mdp_solver,
                    model_class=FLAGS.model_class,
                    num_episodes=FLAGS.num_episodes,
                    num_test_episodes=FLAGS.num_test_episodes,
                    max_len=FLAGS.max_len,
                    log_period=FLAGS.log_period,
                    verbose=FLAGS.verbose,
                )


if __name__ == '__main__':
    app.run(main)
