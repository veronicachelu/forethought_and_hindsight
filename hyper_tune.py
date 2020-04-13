from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm

import prediction_agents
import hypertune_experiment
import prediction_network
import utils
from utils import *
import csv
import copy
from hyperopt import fmin, tpe, Trials
import configs

flags.DEFINE_string('agent', 'vanilla', 'what agent to run')
flags.DEFINE_string('env', 'repeat',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('num_units', 0, 'number of units per hidden layer')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('planning_depth', 1, 'Planning depth for MCTS')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
FLAGS = flags.FLAGS

def get_env(nrng, space):
    if space["env_config"]["non_gridworld"]:
        env_class = getattr(env_utils, space["env_config"]["class"])
        env = env_class(rng=nrng,
                      nS=space["env_config"]["env_size"],
                      obs_type=space["env_config"]["obs_type"]
                      )
        mdp_solver = ChainSolver(env, space["env_config"]["env_size"],
                                 space["env_config"]["nA"], FLAGS.discount)
        env._true_v = mdp_solver.get_optimal_v()
        nS = env._nS
        policy = lambda x: nrng.choice(range(env._nA), p=env._nA * [1 / env._nA])
    else:
        env_class = getattr(env_utils, space["env_config"]["class"])
        env = env_class(path=space["env_config"]["mdp_filename"],
                        stochastic=space["env_config"]["stochastic"],
                        rng=nrng,
                        obs_type=space["env_config"]["obs_type"],
                        env_size=space["env_config"]["env_size"],)
        nS = env._nS
        mdp_solver = MdpSolver(env, nS, space["env_config"]["nA"], FLAGS.discount)
        pi = mdp_solver.get_optimal_policy()
        policy = lambda x: np.argmax(pi[x])
        env._true_v = mdp_solver.get_optimal_v()

    nA = env.action_spec().num_values
    input_dim = env.observation_spec().shape

    return env, nS, nA, input_dim, policy, mdp_solver

def get_agent(env, seed, nrng, nA, input_dim, policy, space):
    rng = jrandom.PRNGKey(seed=seed)
    rng_q, rng_model, rng_agent = jrandom.split(rng, 3)
    network = prediction_network.get_network(
        num_hidden_layers=FLAGS.num_hidden_layers,
        num_units=FLAGS.num_units,
        nA=nA,
        input_dim=input_dim,
        rng=rng_model,
        model_class=space["env_config"]["model_class"])

    agent_class = getattr(prediction_agents,
                          space["agent_config"]["class"][space["env_config"]["model_class"]])

    agent = agent_class(
        run_mode=space["agent_config"]["run_mode"],
        policy=policy,
        action_spec=env.action_spec(),
        network=network,
        batch_size=FLAGS.batch_size,
        discount=FLAGS.discount,
        replay_capacity=space["crt_config"]["replay_capacity"],
        min_replay_size=FLAGS.min_replay_size,
        model_learning_period=FLAGS.model_learning_period,
        planning_iter=space["agent_config"]["planning_iter"],
        planning_period=FLAGS.planning_period,
        planning_depth=space["crt_config"]["planning_depth"],
        lr=space["crt_config"]["lr"],
        lr_model=space["crt_config"]["lr_m"],
        lr_planning=space["crt_config"]["lr"],
        exploration_decay_period=space["env_config"]["num_episodes"],
        seed=seed,
        rng=rng_agent,
        nrng=nrng,
        logs=None,
        max_len=FLAGS.max_len,
        log_period=FLAGS.log_period,
        input_dim=input_dim,
        double_input_reward_model=True
    )
    return agent

def run_experiment(seed, space):
    nrng = np.random.RandomState(seed)
    env, nS, nA, input_dim, policy, mdp_solver = get_env(nrng, space)
    agent = get_agent(env, seed, nrng, nA, input_dim, policy, space)
    return env, agent, mdp_solver

def main(argv):
    hyperparam_folder = os.path.join(os.path.join(FLAGS.logs, "hyper"))
    hyperparam_folder = os.path.join(hyperparam_folder, FLAGS.env)
    hyperparam_folder = os.path.join(hyperparam_folder, FLAGS.agent)
    if not os.path.exists(hyperparam_folder):
        os.makedirs(hyperparam_folder)
    interm_hyperparam_file = os.path.join(hyperparam_folder, "interm_hyperparams.csv")
    final_hyperparam_file = os.path.join(hyperparam_folder, "final_hyperparams.csv")

    persistent_agent_config = configs.agent_config.config[FLAGS.agent]
    if FLAGS.env == "repeat":
        env_config = configs.repeat_config.env_config
        volatile_agent_config = configs.repeat_config.volatile_agent_config
    elif FLAGS.env == "loop":
        env_config = configs.loop_config.env_config
        volatile_agent_config = configs.loop_config.volatile_agent_config
    elif FLAGS.env == "random":
        env_config = configs.random_config.env_config
        volatile_agent_config = configs.random_config.volatile_agent_config
    elif FLAGS.env == "shortcut":
        env_config = configs.shortcut_config.env_config
        volatile_agent_config = configs.shortcut_config.volatile_agent_config
    elif FLAGS.env == "maze":
        env_config = configs.maze_config.env_config
        volatile_agent_config = configs.maze_config.volatile_agent_config
    elif FLAGS.env == "medium_maze":
        env_config = configs.medium_maze_config.env_config
        volatile_agent_config = configs.medium_maze_config.volatile_agent_config

    interm_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    interm_fieldnames.extend(["seed", "avg_steps_seed", 'rmsve_seed'])

    final_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    final_fieldnames.extend(["avg_steps", 'rmsve_avg'])

    if not os.path.exists(interm_hyperparam_file):
        with open(interm_hyperparam_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
            writer.writeheader()
        with open(final_hyperparam_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames)
            writer.writeheader()

    volatile_to_run = list(zip(volatile_agent_config[FLAGS.agent]["planning_depth"],
        volatile_agent_config[FLAGS.agent]["replay_capacity"],
        volatile_agent_config[FLAGS.agent]["lr"],
        volatile_agent_config[FLAGS.agent]["lr_m"],
        ))

    for planning_depth, replay_capacity, lr, lr_m in volatile_to_run:
        seed_config = {"planning_depth": planning_depth,
                      "replay_capacity": replay_capacity,
                      "lr": lr,
                      "lr_m": lr_m}
        final_config = copy.deepcopy(seed_config)
        attributes = list(seed_config.keys())
        attributes.append("seed")

        final_attributes = list(final_config.keys())

        for seed in tqdm(range(0, env_config["num_runs"])):
            seed_config["seed"] = seed
            if not configuration_exists(interm_fieldnames, seed_config, attributes):
                space = {
                    "env_config": env_config,
                    "agent_config": persistent_agent_config,
                    "crt_config": seed_config}

                total_rmsve, avg_steps = run_objective(space)

                with open(interm_fieldnames, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                    seed_config["total_rmsve"] = total_rmsve
                    seed_config["avg_steps"] = avg_steps
                    writer.writerow(seed_config)

        if not configuration_exists(final_fieldnames, final_config, final_attributes):
            rmsve_avg, steps_avg = get_avg_over_seeds(interm_fieldnames, final_config, final_attributes)
            with open(final_fieldnames, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                final_config["rmsve_avg"] = rmsve_avg
                final_config["steps_avg"] = steps_avg
                writer.writerow(final_config)


def get_avg_over_seeds(interm_hyperparam_file, final_config, final_attributes):
    rmsve_avg = []
    steps_avg = []
    with open(interm_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in final_attributes:
                if float(row[key]) != float(final_config[key]):
                    ok = False
                    break
            if ok == True:
                rmsve_avg.append(row['rmsve_seed'])
                steps_avg.append(row['avg_steps_seed'])
        return np.mean(rmsve_avg), np.mean(steps_avg, dtype=int)

def configuration_exists(hyperparam_file, crt_config, attributes):
    with open(hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in attributes:
                if float(row[key]) != float(crt_config[key]):
                    ok = False
                    break
            if ok == True:
                return True
        return False

def run_objective(space):
    seed = space["crt_config"]["seed"]
    if space["env_config"]["non_gridworld"]:
        env, agent, _ = run_experiment(seed, space)
        total_rmsve, avg_steps = hypertune_experiment.run_chain(
            agent=agent,
            model_class=space["env_config"]["model_class"],
            environment=env,
            num_episodes=space["env_config"]["num_episodes"],
        )
        return total_rmsve, avg_steps
    else:
        env, agent, mdp_solver = run_experiment(seed, space)
        total_rmsve, avg_steps = hypertune_experiment.run_episodic(
            agent=agent,
            environment=env,
            mdp_solver=mdp_solver,
            model_class=space["env_config"]["model_class"],
            num_episodes=space["env_config"]["num_episodes"],
            max_len=FLAGS.max_len,
        )
        return total_rmsve, avg_steps

if __name__ == '__main__':
    app.run(main)
