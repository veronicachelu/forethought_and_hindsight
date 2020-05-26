from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm
from main_utils import *
import agents
import experiment
import control_experiment
import network
import utils
from utils import *

# flags.DEFINE_string('agent', 'p_bw_q', 'what agent to run')
# flags.DEFINE_string('agent', 'c_true_bw_q', 'what agent to run')
# flags.DEFINE_string('agent', 'p_fw_q', 'what agent to run')
flags.DEFINE_string('agent', 'q', 'what agent to run')
flags.DEFINE_string('env', 'linear_maze', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 400, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
# flags.DEFINE_integer('planning_depth', 0, 'Planning depthS')
flags.DEFINE_integer('planning_depth', 1, 'Planning depthS')
flags.DEFINE_integer('replay_capacity', 0, 'Replay capacity')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_integer('top_n', 0, 'size of batches sampled from replay')
flags.DEFINE_float('discount', 0.9, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
# flags.DEFINE_float('lr', 0.4, 'learning rate for q optimizer')
flags.DEFINE_float('lr_ctrl', 0.01, 'learning rate for q optimizer')
# flags.DEFINE_float('lr_p', 0.01, 'learning rate for q optimizer')
flags.DEFINE_float('lr_m',  0.01, 'learning rate for model optimizer')
flags.DEFINE_bool('ignore_existent',  True, 'learning rate for model optimizer')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "control")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)

    if not os.path.exists(logs):
        os.makedirs(logs)

    persistent_agent_config = configs.agent_config.config[FLAGS.agent]
    env_config, _ = load_env_and_volatile_configs(FLAGS.env)

    seed_config = {
                    "agent": FLAGS.agent,
                    "planning_depth": FLAGS.planning_depth,
                   "replay_capacity": FLAGS.replay_capacity,
                   # "lr": FLAGS.lr,
                    "top_n": FLAGS.top_n,
                   "lr_ctrl": FLAGS.lr_ctrl,
                   "lr_m": FLAGS.lr_m,}
                   # "lr_p": FLAGS.lr_m}


    for seed in range(0, env_config["num_runs"]):
    # for seed in tqdm(range(0, env_config["num_runs"])):
        seed_config["seed"] = seed
        space = {
            "discount": FLAGS.discount,
            "logs": logs,
            "plot_errors": True,
            "plot_values": True,
            "plot_curves": True,
            "log_period": FLAGS.log_period,
            "env_config": env_config,
            "agent_config": persistent_agent_config,
            "crt_config": seed_config}

        if FLAGS.ignore_existent:
            if (FLAGS.agent == "c_bw_q" or FLAGS.agent == "p_bw_q") and FLAGS.top_n > 0:
                agent_run_mode = "{}_{}".format(FLAGS.agent, FLAGS.top_n)
            else:
                agent_run_mode = "{}".format(FLAGS.agent)
            agent_logs = os.path.join(logs, '{}/summaries/'.format(agent_run_mode))
            agent_logs_seed = os.path.join(agent_logs, 'seed_{}'.format(seed))
            if os.path.exists(agent_logs_seed) and \
                            len(os.listdir(agent_logs_seed)) > 0:
                continue

        run_objective(space)

def run_objective(space):
    aux_agent_configs = {"batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len,
                         "log_period": FLAGS.log_period}
    if space["crt_config"]["agent"].split("_")[0] == "mb":
        aux_agent_configs["pivot"] = space["crt_config"]["agent"].split("_")[1]
    else:
        aux_agent_configs["pivot"] = space["crt_config"]["agent"].split("_")[0]

    seed = space["crt_config"]["seed"]

    env, agent = run_control_experiment(seed, space, aux_agent_configs)

    reward, steps = control_experiment.run_episodic(
        agent=agent,
        environment=env,
        num_episodes=space["env_config"]["control_num_episodes"],
        max_len=FLAGS.max_len,
        space=space,
        aux_agent_configs=aux_agent_configs
    )
    print(reward, steps)

    # reward, steps = control_experiment.test_agent(
    #     agent=agent,
    #     environment=env,
    #     num_episodes=space["env_config"]["control_num_episodes"],
    #     max_len=FLAGS.max_len
    # )
    # print(reward, steps)

    # env, agent = run_control_experiment(seed, space, aux_agent_configs)
    # reward, steps = control_experiment.test_agent(
    #     agent=agent,
    #     environment=env,
    #     num_episodes=space["env_config"]["control_num_episodes"],
    #     max_len=FLAGS.max_len
    # )
    # print(reward, steps)


if __name__ == '__main__':
    app.run(main)
