from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
import configs
import agents
import control_agents
import experiment
import utils
from utils import env_utils
from utils import *
from utils.mdp_solvers.solve_gym import *
from network import *
import haiku as hk
from copy import deepcopy

def get_tabular_chain_env(nrng, space, aux_agent_configs):
    env_class = getattr(env_utils, space["env_config"]["class"])
    env = env_class(rng=nrng,
                    nS=space["env_config"]["env_size"],
                    obs_type=space["env_config"]["obs_type"]
                    )
    mdp_solver = ChainSolver(env, space["env_config"]["env_size"],
                             space["env_config"]["nA"], aux_agent_configs["discount"])
    env._true_v = mdp_solver.get_optimal_v()
    nS = env._nS
    policy = lambda x, nrng: nrng.choice(range(env._nA), p=env._nA * [1 / env._nA])

    return env, nS, policy, mdp_solver

def get_linear_chain_env(nrng, space, aux_agent_configs):
    env_class = getattr(env_utils, space["env_config"]["class"])
    env = env_class(rng=nrng,
                    nS=space["env_config"]["env_size"],
                    nF=space["env_config"]["obs_size"],
                    obs_type=space["env_config"]["obs_type"]
                    )
    mdp_solver = ChainSolver(env, space["env_config"]["env_size"],
                             space["env_config"]["nA"], aux_agent_configs["discount"])
    env._true_v = mdp_solver.get_optimal_v()
    nS = env._nS
    policy = lambda x, nrng: nrng.choice(range(env._nA), p=env._nA * [1 / env._nA])

    return env, nS, policy, mdp_solver

def get_gridworld_env(nrng, space, aux_agent_configs):
    env_class = getattr(env_utils, space["env_config"]["class"])
    env = env_class(path=space["env_config"]["mdp_filename"],
                    stochastic=space["env_config"]["stochastic"],
                    rng=nrng,
                    obs_type=space["env_config"]["obs_type"],
                    env_size=space["env_config"]["env_size"],
                    )
    nS = env._nS
    mdp_solver = MdpSolver(env, nS, space["env_config"]["nA"], aux_agent_configs["discount"])
    if space["env_config"]["policy_type"] == "greedy":
        pi = mdp_solver.get_optimal_policy()
        policy = lambda x, nrng: np.argmax(pi[x])
    else:
        pi = mdp_solver.get_optimal_policy()
        max_indices = np.argmax(pi, -1)
        pi[np.arange(env._nS), :] = space["env_config"]["epsilon"] / space["env_config"]["nA"]
        pi[np.arange(env._nS), max_indices] += 1 - space["env_config"]["epsilon"]
        mdp_solver._pi = pi
        policy = lambda x, nrng: nrng.choice(range(env._nA), p=pi[x])
    env._true_v = mdp_solver.get_optimal_v()

    return env, nS, policy, mdp_solver

def get_continuous_gridworld_env(nrng, space, aux_agent_configs):
    env_class = getattr(env_utils, space["env_config"]["class"])
    env = env_class(path=space["env_config"]["mdp_filename"],
                    stochastic=space["env_config"]["stochastic"],
                    rng=nrng,
                    obs_type=space["env_config"]["obs_type"],
                    env_size=space["env_config"]["env_size"]
                    )
    mdp_solver = MdpSolver(env, None, space["env_config"]["nA"],
                           aux_agent_configs["discount"],
                           feature_coder=space["env_config"]["feature_coder"])
    if space["env_config"]["policy_type"] == "greedy":
        pi = mdp_solver.get_optimal_policy()
        policy = lambda x, nrng: np.argmax(pi[x])
    nS = mdp_solver._nS
    env._true_v = mdp_solver.get_optimal_v()

    return env, nS, policy, mdp_solver

def get_continuous_gym_env(nrng, seed, space, aux_agent_configs):
    env_class = getattr(env_utils, space["env_config"]["class"])
    env = env_class(game=space["env_config"]["mdp_filename"], seed=seed)
    input_dim = env.observation_spec().shape
    nA = env.action_spec().num_values
    gym_solver = GymSolver(env=env, nA=nA, input_dim=input_dim, space=space,
                           seed=seed,
                           aux_agent_configs=aux_agent_configs,
                           nrng=nrng)
    nS = gym_solver._nS
    if space["env_config"]["policy_type"] == "estimated":
        policy = gym_solver.get_optimal_policy()
        v = gym_solver.get_estimated_v()

    return env, nS, policy, gym_solver

def get_env(nrng, seed, space, aux_agent_configs):
    if space["env_config"]["non_gridworld"]:
        if space["env_config"]["model_class"] == "tabular":
            env, nS, policy, mdp_solver = get_tabular_chain_env(nrng, space, aux_agent_configs)
        else:
            if space["env_config"]["env_type"] == "discrete":
                env, nS, policy, mdp_solver = get_linear_chain_env(nrng, space, aux_agent_configs)
            else:
                env, nS, policy, mdp_solver = get_continuous_gym_env(nrng, seed, space, aux_agent_configs)
    else:
        if space["env_config"]["env_type"] == "discrete":
            env, nS, policy, mdp_solver = get_gridworld_env(nrng, space, aux_agent_configs)
        else:
            env, nS, policy, mdp_solver = get_continuous_gridworld_env(nrng, space, aux_agent_configs)

    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape

    return env, nS, nA, input_dim, policy, mdp_solver

def get_control_env(nrng, seed, space, aux_agent_configs):
    if space["env_config"]["env_type"] == "discrete":
        env_class = getattr(env_utils, space["env_config"]["class"])
        env = env_class(path=space["env_config"]["mdp_filename"],
                        stochastic=space["env_config"]["stochastic"],
                        rng=nrng,
                        obs_type=space["env_config"]["obs_type"],
                        env_size=space["env_config"]["env_size"],
                        )
        nS = env._nS
    else:
        env_class = getattr(env_utils, space["env_config"]["class"])
        env = env_class(game=space["env_config"]["mdp_filename"], seed=seed)
        input_dim = env.observation_spec().shape
        nS = np.prod(input_dim)

    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape

    return env, nS, nA, input_dim

# def get_control_agent(env, seed, nrng, nA, input_dim, space):
#     rng = jrandom.PRNGKey(seed=seed)
#     rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
#     rng_sequence = hk.PRNGSequence(rng_agent)
#     network = get_network(
#         pg=space["agent_config"]["pg"],
#         num_hidden_layers=space["agent_config"]["num_hidden_layers"],
#         num_units=space["agent_config"]["num_units"],
#         nA=nA,
#         input_dim=input_dim,
#         rng=rng_model,
#         rng_target=rng_q,
#         feature_coder=space["env_config"]["feature_coder"],
#         latent=space["agent_config"]["latent"],
#         model_family="q",
#         model_class=space["env_config"]["model_class"],
#         target_networks=space["agent_config"]["target_networks"])
#
#     agent = VanillaQ(run_mode='q',
#                      action_spec=env.action_spec(),
#                      network=network,
#                      batch_size=1,
#                      discount=0.99,
#                      lr=space["crt_config"]["lr"],
#                      exploration_decay_period=space["env_config"]["num_episodes"],
#                      nrng=nrng,
#                      seed=seed,
#                      logs=space["logs"],
#                      log_period=space["log_period"],
#                      latent=space["agent_config"]["latent"],
#                      feature_coder=space["env_config"]["feature_coder"],
#                      target_networks=space["agent_config"]["target_networks"]
#                      )
#
#     return agent

def get_control_agent(env, seed, nrng, nA, input_dim, space):
    rng = jrandom.PRNGKey(seed=seed)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    rng_sequence = hk.PRNGSequence(rng_agent)
    if space["agent_config"]["task_type"] == "prediction":
        control_space = deepcopy(space)
        control_space["agent_config"] = \
            configs.agent_config.config[space["agent_config"]["control_agent"]]
    else:
        control_space = deepcopy(space)

    network = get_network(
        pg=control_space["agent_config"]["pg"],
        num_hidden_layers=control_space["agent_config"]["num_hidden_layers"],
        num_units=control_space["agent_config"]["num_units"],
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        rng_target=rng_q,
        feature_coder=control_space["env_config"]["feature_coder"],
        latent=control_space["agent_config"]["latent"],
        model_family=control_space["agent_config"]["model_family"],
        model_class=control_space["env_config"]["model_class"],
        target_networks=control_space["agent_config"]["target_networks"])

    agent_class = getattr(control_agents,
                          control_space["agent_config"]["class"][control_space["env_config"]["model_class"]])

    agent = agent_class(
        run_mode=control_space["agent_config"]["run_mode"],
        action_spec=env.action_spec(),
        network=network,
        batch_size=1,
        discount=0.99,
        lr=control_space["crt_config"]["lr_ctrl"],
        exploration_decay_period=control_space["env_config"]["num_episodes"],
        seed=seed,
        rng_seq=rng_sequence,
        nrng=nrng,
        logs=control_space["logs"],
        log_period=control_space["log_period"],
        latent=control_space["agent_config"]["latent"],
        feature_coder=control_space["env_config"]["feature_coder"],
        target_networks=control_space["agent_config"]["target_networks"]
    )
    return agent

def get_agent(env, seed, nrng, nA, input_dim, policy, space, aux_agent_configs):
    rng = jrandom.PRNGKey(seed=seed)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    rng_sequence = hk.PRNGSequence(rng_agent)
    network = get_network(
        pg=space["agent_config"]["pg"],
        num_hidden_layers=space["agent_config"]["num_hidden_layers"],
        num_units=space["agent_config"]["num_units"],
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        rng_target=rng_q,
        feature_coder=space["env_config"]["feature_coder"],
        latent=space["agent_config"]["latent"],
        model_family=space["agent_config"]["model_family"],
        model_class=space["env_config"]["model_class"],
        target_networks=space["agent_config"]["target_networks"])

    agent_class = getattr(agents,
                          space["agent_config"]["class"][space["env_config"]["model_class"]])

    agent = agent_class(
        run_mode=space["agent_config"]["run_mode"],
        policy=policy,
        policy_type=space["env_config"]["policy_type"],
        action_spec=env.action_spec(),
        network=network,
        batch_size=aux_agent_configs["batch_size"],
        discount=aux_agent_configs["discount"],
        replay_capacity=space["crt_config"]["replay_capacity"],
        min_replay_size=aux_agent_configs["min_replay_size"],
        model_learning_period=aux_agent_configs["model_learning_period"],
        planning_iter=space["agent_config"]["planning_iter"],
        planning_period=aux_agent_configs["planning_period"],
        planning_depth=space["crt_config"]["planning_depth"],
        lr=space["crt_config"]["lr"],
        lr_model=space["crt_config"]["lr_m"],
        lr_planning=space["crt_config"]["lr_p"],
        exploration_decay_period=space["env_config"]["num_episodes"],
        seed=seed,
        nrng=nrng,
        rng=rng_sequence,
        logs=space["logs"],
        max_len=aux_agent_configs["max_len"],
        log_period=space["log_period"],
        input_dim=input_dim,
        latent=space["agent_config"]["latent"],
        feature_coder=space["env_config"]["feature_coder"],
        target_networks=space["agent_config"]["target_networks"]
        # double_input_reward_model=True
    )
    return agent

def run_experiment(seed, space, aux_agent_configs):
    nrng = np.random.RandomState(seed)
    env, nS, nA, input_dim, policy, mdp_solver = get_env(nrng, seed, space, aux_agent_configs)
    agent = get_agent(env, seed, nrng, nA, input_dim, policy, space, aux_agent_configs)
    return env, agent, mdp_solver

def run_control_experiment(seed, space, aux_agent_configs):
    nrng = np.random.RandomState(seed)
    env, nS, nA, input_dim = get_control_env(nrng, seed, space, aux_agent_configs)
    agent = get_control_agent(env, seed, nrng, nA, input_dim, space)
    return env, agent

def load_env_and_volatile_configs(env):
    if env == "repeat":
        env_config = configs.repeat_config.env_config
        volatile_agent_config = configs.repeat_config.volatile_agent_config
    elif env == "loop":
        env_config = configs.loop_config.env_config
        volatile_agent_config = configs.loop_config.volatile_agent_config
    elif env == "random":
        env_config = configs.random_config.env_config
        volatile_agent_config = configs.random_config.volatile_agent_config
    elif env == "shortcut":
        env_config = configs.shortcut_config.env_config
        volatile_agent_config = configs.shortcut_config.volatile_agent_config
    elif env == "maze":
        env_config = configs.maze_config.env_config
        volatile_agent_config = configs.maze_config.volatile_agent_config
    elif env == "linear_maze":
        env_config = configs.linear_maze_config.env_config
        volatile_agent_config = configs.linear_maze_config.volatile_agent_config
    elif env == "linear_tiny_maze":
        env_config = configs.linear_tiny_maze_config.env_config
        volatile_agent_config = configs.linear_tiny_maze_config.volatile_agent_config
    elif env == "random_maze":
        env_config = configs.random_maze_config.env_config
        volatile_agent_config = configs.random_maze_config.volatile_agent_config
    elif env == "medium_maze":
        env_config = configs.medium_maze_config.env_config
        volatile_agent_config = configs.medium_maze_config.volatile_agent_config
    elif env == "random_medium_maze":
        env_config = configs.random_medium_maze_config.env_config
        volatile_agent_config = configs.random_medium_maze_config.volatile_agent_config
    elif env == "boyan":
        env_config = configs.boyan_config.env_config
        volatile_agent_config = configs.boyan_config.volatile_agent_config
    elif env == "puddle":
        env_config = configs.puddle_config.env_config
        volatile_agent_config = configs.puddle_config.volatile_agent_config
    elif env == "obstacle":
        env_config = configs.obstacle_config.env_config
        volatile_agent_config = configs.obstacle_config.volatile_agent_config
    elif env == "cartpole":
        env_config = configs.cartpole_config.env_config
        volatile_agent_config = configs.cartpole_config.volatile_agent_config
    elif env == "random_linear":
        env_config = configs.random_linear_config.env_config
        volatile_agent_config = configs.random_linear_config.volatile_agent_config
    elif env == "stoch_linear_maze":
        env_config = configs.stoch_linear_maze_config.env_config
        volatile_agent_config = configs.stoch_linear_maze_config.volatile_agent_config


    return env_config, volatile_agent_config

def build_hyper_list(agent, volatile_agent_config, up_to=None):
    volatile_to_run = []
    limited_volatile_to_run = []
    for planning_depth in volatile_agent_config[agent]["planning_depth"]:
        if up_to is not None and planning_depth > up_to:
            break
        for replay_capacity in volatile_agent_config[agent]["replay_capacity"]:
            limited_volatile_to_run.append([planning_depth, replay_capacity])
            for lr in volatile_agent_config[agent]["lr"]:
                for lr_p in volatile_agent_config[agent]["lr_p"]:
                    for lr_m in volatile_agent_config[agent]["lr_m"]:
                        volatile_to_run.append([planning_depth, replay_capacity,
                                                round(lr, 3), round(lr_p, 3), round(lr_m, 3),
                                                round(volatile_agent_config[agent]["lr_ctrl"], 3)])
    return limited_volatile_to_run, volatile_to_run