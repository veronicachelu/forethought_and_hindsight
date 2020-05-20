from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
from main_utils import *
import agents
import experiment
import network
import utils
from run_utils import *
from utils import *

# flags.DEFINE_string('agent', 'mb_c_bw_PAML', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_c_bwfw', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_c_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_p_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_c_true_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_p_true_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'mb_p_fw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_bwfw', 'what agent to run')


# flags.DEFINE_string('agent', 'vanilla', 'what agent to run')
# flags.DEFINE_string('agent', 'c_random_bw', 'what agent to run')


# flags.DEFINE_string('agent', 'p_fw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_fw', 'what agent to run')
# flags.DEFINE_string('agent', 'p_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_bw', 'what agent to run')

# flags.DEFINE_string('agent', 'p_true_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'p_fw_PAML', 'what agent to run')
flags.DEFINE_string('agent', 'p_bw_PAML', 'what agent to run')
# flags.DEFINE_string('agent', 'c_bw_PAML', 'what agent to run')
flags.DEFINE_string('env', 'bipartite_linear', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 10000000, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 0, 'Number of timesteps of real experience to see before running planning')
# flags.DEFINE_integer('planning_depth', 0, 'Planning depth')
flags.DEFINE_integer('planning_depth', 1, 'Planning depth')
flags.DEFINE_integer('replay_capacity', 0, 'Replay capacity')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .9, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
flags.DEFINE_float('lr', 0.1, 'learning rate for q optimizer')
flags.DEFINE_float('lr_ctrl', 0.4, 'learning rate for q optimizer')
flags.DEFINE_float('lr_p', 0.1, 'learning rate for q optimizer')
flags.DEFINE_float('lr_m',  0.1, 'learning rate for model optimizer')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)

    if not os.path.exists(logs):
        os.makedirs(logs)

    agent = {
        "agent": FLAGS.agent,
        "planning_depth": FLAGS.planning_depth,
        "replay_capacity": FLAGS.replay_capacity,
        "lr": FLAGS.lr,
        "lr_m": FLAGS.lr_m,
        "lr_p": FLAGS.lr_p,
        "lr_ctrl": FLAGS.lr_ctrl,
    }
    aux_agent_configs = {"batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len,
                         "log_period": FLAGS.log_period}
    aux_agent_configs["mb"] = True if agent["agent"].split("_")[0] == "mb" else False
    if agent["agent"].split("_")[0] == "mb":
        aux_agent_configs["pivot"] = agent["agent"].split("_")[1]
    else:
        aux_agent_configs["pivot"] = agent["agent"].split("_")[0]
    run_agent(FLAGS.env, agent, logs, aux_agent_configs, ignore_existent=True)


if __name__ == '__main__':
    app.run(main)
