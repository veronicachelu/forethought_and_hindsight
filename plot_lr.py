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
from run_utils import *
import csv

flags.DEFINE_string('env', 'random_linear', 'env')
flags.DEFINE_string('agent', 'vanilla', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100000, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')

FLAGS = flags.FLAGS

def get_hyperparams():
    hyperparam_folder = os.path.join(FLAGS.logs, "hyper")
    hyperparam_folder = os.path.join(hyperparam_folder, FLAGS.env)

    maze_best_aoc_hyperparams = os.path.join(hyperparam_folder,
                                             "{}_best_aoc_hyperparams.csv".format(FLAGS.env))
    maze_best_min_hyperparams = os.path.join(hyperparam_folder,
                                             "{}_best_min_hyperparams.csv".format(FLAGS.env))

    agents = []
    with open(maze_best_aoc_hyperparams, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for agent in reader:
            agent["lr_ctrl"] = 0.4
            agents.append(agent)

    return agents

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)

    if not os.path.exists(logs):
        os.makedirs(logs)

    agents = get_hyperparams()

    aux_agent_configs = {"batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len,
                         "log_period": FLAGS.log_period}
