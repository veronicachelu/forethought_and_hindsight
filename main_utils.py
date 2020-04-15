from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
import configs
import agents
import experiment
import utils
from utils import *
from network import *

def get_env(nrng, space, aux_agent_configs):
    if space["env_config"]["non_gridworld"]:
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
    else:
        env_class = getattr(env_utils, space["env_config"]["class"])
        env = env_class(path=space["env_config"]["mdp_filename"],
                        stochastic=space["env_config"]["stochastic"],
                        rng=nrng,
                        obs_type=space["env_config"]["obs_type"],
                        env_size=space["env_config"]["env_size"],)
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

    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape

    return env, nS, nA, input_dim, policy, mdp_solver

def get_agent(env, seed, nrng, nA, input_dim, policy, space, aux_agent_configs):
    rng = jrandom.PRNGKey(seed=seed)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    network = get_network(
        num_hidden_layers=aux_agent_configs["num_hidden_layers"],
        num_units=aux_agent_configs["num_units"],
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        model_family=space["agent_config"]["model_family"],
        model_class=space["env_config"]["model_class"],
        target_networks=space["agent_config"]["target_networks"])

    agent_class = getattr(agents,
                          space["agent_config"]["class"][space["env_config"]["model_class"]])

    agent = agent_class(
        run_mode=space["agent_config"]["run_mode"],
        policy=policy,
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
        logs=space["logs"],
        max_len=aux_agent_configs["max_len"],
        log_period=space["log_period"],
        input_dim=input_dim,
        latent=space["agent_config"]["latent"],
        target_networks=space["agent_config"]["target_networks"]
        # double_input_reward_model=True
    )
    return agent

def run_experiment(seed, space, aux_agent_configs):
    nrng = np.random.RandomState(seed)
    env, nS, nA, input_dim, policy, mdp_solver = get_env(nrng, space, aux_agent_configs)
    agent = get_agent(env, seed, nrng, nA, input_dim, policy, space, aux_agent_configs)
    return env, agent, mdp_solver

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
    elif env == "random_maze":
        env_config = configs.random_maze_config.env_config
        volatile_agent_config = configs.random_maze_config.volatile_agent_config
    elif env == "medium_maze":
        env_config = configs.medium_maze_config.env_config
        volatile_agent_config = configs.medium_maze_config.volatile_agent_config
    elif env == "random_medium_maze":
        env_config = configs.random_medium_maze_config.env_config
        volatile_agent_config = configs.random_medium_maze_config.volatile_agent_config

    return env_config, volatile_agent_config

def build_hyper_list(agent, volatile_agent_config):
    volatile_to_run = []
    limited_volatile_to_run = []
    for planning_depth in volatile_agent_config[agent]["planning_depth"]:
        for replay_capacity in volatile_agent_config[agent]["replay_capacity"]:
            limited_volatile_to_run.append([planning_depth, replay_capacity])
            for lr in volatile_agent_config[agent]["lr"]:
                for lr_p in volatile_agent_config[agent]["lr_p"]:
                    for lr_m in volatile_agent_config[agent]["lr_m"]:
                        volatile_to_run.append([planning_depth, replay_capacity,
                                                round(lr, 2), round(lr_p, 2), round(lr_m, 2)])
    return limited_volatile_to_run, volatile_to_run