from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
from main_utils import *
import agents
import experiment
import network
import utils
from utils import *
import csv

flags.DEFINE_string('env', 'maze', 'env')
# flags.DEFINE_string('env', 'linear_maze', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100000, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')

FLAGS = flags.FLAGS


def get_hyperparams():
    hyperparam_folder = os.path.join(FLAGS.logs, "hyper")
    hyperparam_folder = os.path.join(hyperparam_folder, FLAGS.env)

    maze_best_aoc_hyperparams = os.path.join(hyperparam_folder, "maze_best_aoc_hyperparams.csv")
    maze_best_min_hyperparams = os.path.join(hyperparam_folder, "maze_best_min_hyperparams.csv")

    agents = []
    with open(maze_best_aoc_hyperparams, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for agent in reader:
            agents.append(agent)

    return agents

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)

    if not os.path.exists(logs):
        os.makedirs(logs)

    agents = get_hyperparams()

    for agent in agents:
        run_agent(agent, logs)

def run_agent(agent, logs):
    persistent_agent_config = configs.agent_config.config[agent["agent"]]
    agent_run_mode = "{}_{}_{}".format(persistent_agent_config["run_mode"], agent["planning_depth"], agent["replay_capacity"])
    agent_logs = os.path.join(logs, '{}/summaries/'.format(agent_run_mode))
    if os.path.exists(agent_logs):
        return

    env_config, _ = load_env_and_volatile_configs(FLAGS.env)

    seed_config = {"planning_depth": int(agent["planning_depth"]),
                   "replay_capacity": int(agent["replay_capacity"]),
                   "lr": float(agent["lr"]),
                   "lr_m": float(agent["lr_m"]),
                   "lr_p": float(agent["lr_m"])}

    for seed in tqdm(range(0, env_config["num_runs"])):
        seed_config["seed"] = seed
        space = {
            "logs": logs,
            "plot_errors": True,
            "plot_values": True,
            "plot_curves": True,
            "log_period": FLAGS.log_period,
            "env_config": env_config,
            "agent_config": persistent_agent_config,
            "crt_config": seed_config}

        run_objective(space)


def run_objective(space):
    aux_agent_configs = {
                         "batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len,
                         "log_period": FLAGS.log_period}
    seed = space["crt_config"]["seed"]
    env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)

    if space["env_config"]["non_gridworld"]:
        experiment.run_chain(
            agent=agent,
            environment=env,
            mdp_solver=mdp_solver,
            model_class=space["env_config"]["model_class"],
            num_episodes=space["env_config"]["num_episodes"],
            plot_curves=space["plot_curves"],
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            log_period=space["log_period"],
        )
    else:
        experiment.run_episodic(
            agent=agent,
            environment=env,
            mdp_solver=mdp_solver,
            model_class=space["env_config"]["model_class"],
            num_episodes=space["env_config"]["num_episodes"],
            max_len=FLAGS.max_len,
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            plot_curves=space["plot_curves"],
            log_period=space["log_period"],
        )


if __name__ == '__main__':
    app.run(main)
