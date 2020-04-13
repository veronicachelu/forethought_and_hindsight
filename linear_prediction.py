from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm

import prediction_agents
import prediction_experiment
import prediction_network
import utils
from utils import *

# flags.DEFINE_string('run_mode', 'vanilla', 'what agent to run')
flags.DEFINE_string('run_mode', 'explicit_v', 'what agent to run')
flags.DEFINE_boolean('optimal_policy', True, 'optimal_policy')
# flags.DEFINE_boolean('no_latent', False, 'no_latent')
flags.DEFINE_boolean('no_latent', True, 'no_latent')
flags.DEFINE_string('policy', 'optimal', 'optimal or random')
flags.DEFINE_string('model_class', 'linear', 'tabular or linear')
flags.DEFINE_string('model_family', 'intrinsic', 'tabular or linear')
# flags.DEFINE_string('model_family', 'extrinsic', 'tabular or linear')
# flags.DEFINE_string('model_class', 'tabular', 'tabular or linear')
# flags.DEFINE_string('env_type', 'continuous', 'discrete or continuous')
flags.DEFINE_string('env_type', 'discrete', 'discrete or continuous')
flags.DEFINE_string('obs_type', 'onehot', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('obs_type', 'tile', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('obs_type', 'tabular', 'onehot, tabular, tile for continuous')
flags.DEFINE_integer('max_reward', 1, 'max reward')
# flags.DEFINE_string('mdp', './continuous_mdps/obstacle.mdp',
# flags.DEFINE_string('mdp', './mdps/simple.mdp',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
# flags.DEFINE_string('mdp', 'boyan_chain',
# flags.DEFINE_string('mdp', 'random_chain',
# flags.DEFINE_string('mdp', 'loopy_chain',
# flags.DEFINE_string('mdp', 'po',
# flags.DEFINE_string('mdp', 'repeat',
# flags.DEFINE_string('mdp', 'shortcut',
# flags.DEFINE_string('mdp', 'serial'   ,
# flags.DEFINE_string('mdp', 'bandit',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
flags.DEFINE_string('mdp', './mdps/maze_48.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_80.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_221.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_486.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_864.mdp',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_integer('env_size', 1, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
# flags.DEFINE_integer('env_size', 1, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
# flags.DEFINE_integer('env_size', 5, 'Discreate - Env size: 1x, 2x, 4x, 10x, but without the x.'
                                    'Continuous - Num of bins for each dimension of the discretization')
# flags.DEFINE_integer('n_hidden_states', 6, 'max reward')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
# flags.DEFINE_integer('num_episodes', 1800, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 180, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_episodes', 70, 'Number of episodes to run for.')
flags.DEFINE_integer('num_episodes', 100, 'Number of episodes to run for.')
flags.DEFINE_integer('num_runs', 10, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_runs', 0, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_steps', 2000, 'Number of episodes to run for.')
# flags.DEFINE_integer('num_steps', 1000, 'Number of episodes to run for.')
flags.DEFINE_integer('num_steps', 300, 'Number of episodes to run for.')
flags.DEFINE_integer('num_test_episodes', 100, 'Number of test episodes to run for.')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
# flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('num_units', 8, 'number of units per hidden layer')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('planning_depth', 1, 'Planning depth for MCTS')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
# flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 50, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
flags.DEFINE_float('lr', 1e-2, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 5e-3, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 1, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 0.2, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 0.2, 'learning rate for q optimizer')
flags.DEFINE_float('lr_planning', 1e-2, 'learning rate for q optimizer')
# flags.DEFINE_float('lr', 1e-3, 'learning rate for q optimizer')
# flags.DEFINE_float('lr_model', 1e-2, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 0.01, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model', 1e-3, 'learning rate for model optimizer')
flags.DEFINE_float('lr_model', 5e-3, 'learning rate for model optimizer')
# flags.DEFINE_float('lr_model',  5e-4, 'learning rate for model optimizer')
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
flags.DEFINE_boolean('double_input_reward_model', True, 'double_input_reward_model or not.')

FLAGS = flags.FLAGS
NON_GRIDWORLD_MDPS = ["random_chain", "boyan_chain", "bandit", "shortcut",
                      "loop", "tree", "repeat", "serial",
                      "po"]

run_mode_to_agent_prop = {
        "vanilla_intrinsic": {"linear":
                        {"class": "LpIntrinsicVanilla"},},
        "explicit_v": {"linear":
                        {"class": "LpExplicitValueBased"},},
        "vanilla": {"linear":
                        {"class": "LpVanilla"},
                    "tabular":
                        {"class": "TpVanilla"},
                    },
        "fw": {"linear":
                        {"class": "LpFw"},
                    "tabular":
                        {"class": "TpFw"},
                    },
        "fw_rnd": {"linear":
                        {"class": "LpFwRnd"},
                    "tabular":
                        {"class": "TpFwRnd"},
                    },
        "fw_pri": {"linear":
                        {"class": "LpFwPri"},
                    "tabular":
                        {"class": "TpFwPri"},
                    },
        "pred_exp": {"linear":
                       {"class": "nStepLpPredExp"},
                   "tabular":
                       {"class": "nStepTpPredDistrib"},
                   },
        "implicit_gen": {"linear":
                       {"class": "LpImplicitGen"},
                   "tabular":
                       {"class": "TpImplicitGen"},
                   },
        "explicit_exp": {"linear":
                      {"class": "LpExplicitExp"},
                  "tabular":
                      {"class": "TpExplicitDistrib"},
                  },
        "explicit_gen": {"linear":
                      {"class": "LpExplicitGen"},
                  "tabular":
                      {"class": "TpExplicitGen"},
                  },
         "explicit_true": {"linear":
                         {"class": "LpExplicitTrue"},
                     "tabular":
                         {"class": "TpExplicitTrue"},
                     },
        "explicit_iterat": {"linear":
                         {"class": "LpExplicitIterat"},
                     "tabular":
                         {"class": "TpExplicitIterat"},
                     },
        "fw_bw_PWMA": {"linear":
                      {"class": "LpFwBwPWMA"},
                  "tabular":
                      {"class": "TpFwBwPWMA"},
                  },
        "fw_bw_MG": {"linear":
                      {"class": "LpFwBwMG"},
                  "tabular":
                      {"class": "TpFwBwMG"},
                  },
        "fw_bw_Imprv": {"linear":
                     {"class": "LpFwBwImprv"},
                 "tabular":
                     {"class": "TpFwBwImprv"},
                 },
        "bw_fw_exp": {"linear":
                      {"class": "LpBwFwExp"},
                  "tabular":
                      {"class": "TpBwFwDistrib"},
                  },
        "bw_fw_gen": {"linear":
                      {"class": "LpBwFwGen"},
                  "tabular":
                      {"class": "TpBwFwGen"},
                  },
    }

def get_env(nrng, logs):
    if FLAGS.mdp == "random_chain":
        env = RandomChain(rng=nrng,
                          nS=FLAGS.env_size,
                          obs_type=FLAGS.obs_type
                          )
        mdp_solver = ChainSolver(env, FLAGS.env_size, 2, FLAGS.discount)
        # policy = mdp_solver.get_optimal_policy()
        env._true_v = mdp_solver.get_optimal_v()
        nS = env._nS
    elif FLAGS.mdp == "boyan_chain":
        env = BoyanChain(rng=nrng,
                          nS=FLAGS.n_hidden_states,
                          nF=FLAGS.env_size,
                          obs_type=FLAGS.obs_type
                          )
        nS = env._nF
    elif FLAGS.mdp == "tree":
        env = Tree(rng=nrng,
                         nA=FLAGS.n_hidden_states,
                         h=FLAGS.env_size,
                         obs_type=FLAGS.obs_type
                         )
        nS = env._nS
        nA = env._nA
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "bandit":
        env = Bandit(rng=nrng)
        nS = env._nS
        nA = env._nA
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "loop":
        env = Loop(rng=nrng, nS=FLAGS.env_size, obs_type=FLAGS.obs_type)
        nS = env._nS
        nA = 1
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "repeat":
        env = Repeat(rng=nrng, nS=FLAGS.env_size, obs_type=FLAGS.obs_type)
        nS = env._nS
        nA = 1
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "po":
        env = PO(rng=nrng, nS=FLAGS.env_size)
        nS = env._nS
        nA = 1
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "shortcut":
        env = Shortcut(rng=nrng, nS=FLAGS.env_size, obs_type=FLAGS.obs_type)
        nS = env._nS
        nA = 2
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
    elif FLAGS.mdp == "serial":
        env = Serial(rng=nrng, nS=FLAGS.env_size, obs_type=FLAGS.obs_type)
        nS = env._nS
        nA = 1
        mdp_solver = ChainSolver(env, nS, nA, FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
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

    if FLAGS.mdp in NON_GRIDWORLD_MDPS:
        policy = lambda x: nrng.choice(range(env._nA), p=env._nA * [1/env._nA])
        mdp_solver = None
    else:
        plot_grid(env, logs, env_type=FLAGS.env_type)
        mdp_solver = MdpSolver(env, nS, nA, FLAGS.discount)
        if FLAGS.optimal_policy:
            pi = mdp_solver.get_optimal_policy()
            policy = lambda x: np.argmax(pi[np.argmax(x)])
        else:
            policy = lambda x: nrng.choice(range(env._nA), p=env._nA * [1 / env._nA])
        v = mdp_solver.get_optimal_v()
        # v = env.reshape_v(v)
        # plot_v(env, v, logs, env_type=FLAGS.env_type)
        # plot_policy(env, env.reshape_pi(mdp_solver._pi), logs, env_type=FLAGS.env_type)
        # eta_pi = mdp_solver.get_eta_pi(mdp_solver._pi)
        # plot_eta_pi(env, env.reshape_v(eta_pi), logs, env_type=FLAGS.env_type)

    return env, nS, nA, input_dim, policy, mdp_solver

def get_agent(env, seed, nrng, nA, input_dim, policy, logs):
    rng = jrandom.PRNGKey(seed=seed)
    # rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    network = prediction_network.get_network(num_hidden_layers=FLAGS.num_hidden_layers,
                                              num_units=FLAGS.num_units,
                                              nA=nA,
                                              input_dim=input_dim,
                                              rng=rng,
                                              model_family=FLAGS.model_family,
                                              model_class=FLAGS.model_class,
                                              double_input_reward_model=True,
                                             )
    agent_prop = run_mode_to_agent_prop[FLAGS.run_mode]
    run_mode = FLAGS.run_mode
    agent_class = getattr(prediction_agents, agent_prop[FLAGS.model_class]["class"])

    agent = agent_class(
        run_mode=run_mode,
        policy=policy,
        action_spec=env.action_spec(),
        # v_network=v_network,
        # v_parameters=v_network_params,
        # model_network=model_network,
        # model_parameters=model_network_params,
        network=network,
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
        nrng=nrng,
        logs=logs,
        no_latent=FLAGS.no_latent,
        max_len=FLAGS.max_len,
        log_period=FLAGS.log_period,
        input_dim=input_dim,
        double_input_reward_model=FLAGS.double_input_reward_model
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

    if FLAGS.mdp in NON_GRIDWORLD_MDPS:
        for seed in tqdm(range(0, FLAGS.num_runs)):
            env, agent, _ = run_experiment(seed, logs)

            prediction_experiment.run_chain(
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
                prediction_experiment.run(
                    agent=agent,
                    environment=env,
                    mdp_solver=mdp_solver,
                    model_class=FLAGS.model_class,
                    num_steps=FLAGS.num_steps,
                    log_period=FLAGS.log_period,
                    verbose=FLAGS.verbose
                )
            else:
                prediction_experiment.run_episodic(
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
