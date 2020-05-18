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


flags.DEFINE_string('agent', 'bw_fw', 'what agent to run')

# flags.DEFINE_string('agent', 'p_fw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_fw', 'what agent to run')
# flags.DEFINE_string('agent', 'p_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_bw', 'what agent to run')

# flags.DEFINE_string('agent', 'p_true_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'c_true_bw', 'what agent to run')
# flags.DEFINE_string('agent', 'p_bw_PAML', 'what agent to run')
# flags.DEFINE_string('agent', 'c_bw_PAML', 'what agent to run')
flags.DEFINE_string('env', 'obstacle', 'env')
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
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
flags.DEFINE_float('lr', 0.001, 'learning rate for q optimizer')
flags.DEFINE_float('lr_ctrl', 0.4, 'learning rate for q optimizer')
flags.DEFINE_float('lr_p', 0.001, 'learning rate for q optimizer')
flags.DEFINE_float('lr_m',  0.01, 'learning rate for model optimizer')

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
    run_agent_gain(FLAGS.env, agent, logs, aux_agent_configs, ignore_existent=False)

def run_agent_gain(env, agent, logs, aux_agent_configs, ignore_existent=False):
    persistent_agent_config = configs.agent_config.config[agent["agent"]]
    agent_run_mode = "{}_{}_{}".format(persistent_agent_config["run_mode"], agent["planning_depth"],
                                       agent["replay_capacity"])
    agent_logs = os.path.join(logs, '{}/summaries/'.format(agent_run_mode))

    env_config, _ = load_env_and_volatile_configs(env)

    seed_config = {"planning_depth": int(agent["planning_depth"]),
                   "replay_capacity": int(agent["replay_capacity"]),
                   "lr": float(agent["lr"]),
                   "lr_m": float(agent["lr_m"]),
                   "lr_ctrl": float(agent["lr_ctrl"]),
                   "lr_p": float(agent["lr_p"])}

    # for seed in tqdm(range(0, env_config["num_runs"])):
    for seed in range(0, env_config["num_runs"]):
        seed_config["seed"] = seed
        space = {
            "logs": logs,
            "plot_errors": True,
            "plot_values": True,
            "plot_curves": True,
            "log_period": aux_agent_configs["log_period"],
            "env_config": env_config,
            "agent_config": persistent_agent_config,
            "crt_config": seed_config}

        if ignore_existent:
            agent_logs_seed = os.path.join(agent_logs, 'seed_{}'.format(seed))
            if os.path.exists(agent_logs_seed) and \
                            len(os.listdir(agent_logs_seed)) > 0:
                continue
        run_objective(space, aux_agent_configs)


def run_objective(space, aux_agent_configs):
    seed = space["crt_config"]["seed"]
    env, agent, mdp_solver = main_utils.run_experiment(seed, space, aux_agent_configs)

    if space["agent_config"]["model_family"] == "true" and \
                    space["env_config"]["model_class"] == "tabular":
        agent._o_network, agent._fw_o_network, agent._r_network, _ = mdp_solver.get_true_model()
    elif space["agent_config"]["model_family"] == "random" and \
                    space["env_config"]["model_class"] == "tabular":
        _, _, agent._r_network, _ = mdp_solver.get_true_model()
    elif space["agent_config"]["model_family"] == "PAML" and \
        space["env_config"]["model_class"] == "tabular":
        _, _, _, agent._true_v_network = mdp_solver.get_true_model()
    total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_gain(
        agent=agent,
        space=space,
        aux_agent_configs=aux_agent_configs,
        mdp_solver=mdp_solver,
        environment=env,
    )

if __name__ == '__main__':
    app.run(main)
