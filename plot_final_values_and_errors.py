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

flags.DEFINE_string('agent', 'vanilla', 'what agent to run')
flags.DEFINE_string('env', 'repeat',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100000, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('num_units', 0, 'number of units per hidden layer')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('planning_depth', 0, 'Planning depth for MCTS')
flags.DEFINE_integer('replay_capacity', 0, 'Planning depth for MCTS')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
flags.DEFINE_float('lr', 2e-1, 'learning rate for q optimizer')
flags.DEFINE_float('lr_p', 2e-1, 'learning rate for q optimizer')
flags.DEFINE_float('lr_m',  2e-1, 'learning rate for model optimizer')

FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)

    if not os.path.exists(logs):
        os.makedirs(logs)

    persistent_agent_config = configs.agent_config.config[FLAGS.agent]
    env_config, _ = load_env_and_volatile_configs(FLAGS.env)

    seed_config = {"planning_depth": FLAGS.planning_depth,
                   "replay_capacity": FLAGS.replay_capacity,
                   "lr": FLAGS.lr,
                   "lr_m": FLAGS.lr_m}

    seed_values = []
    seed_errors = []
    for seed in tqdm(range(0, 10)):
        seed_config["seed"] = seed
        space = {
            "logs": logs,
            "plot_errors": True,
            "plot_values": True,
            "plot_curves": False,
            "log_period": FLAGS.log_period,
            "env_config": env_config,
            "agent_config": persistent_agent_config,
            "crt_config": seed_config}

        _, _, values, errors, env, agent, mdp_solver = run_objective(space)
        seed_values.append(values)
        seed_errors.append(errors)

    avg_error = np.mean(errors, axis=0)
    avg_v = np.mean(values, axis=0)
    plot_error(env=env,
               values=env.reshape_v(avg_error),
               logs=agent._images_dir,
               eta_pi=env.reshape_v(mdp_solver.get_eta_pi(mdp_solver._pi)),
               filename="error_{}.png".format(agent.episode))
    plot_v(env=env,
               values=env.reshape_v(avg_v),
               logs=agent._images_dir,
               true_v=env.reshape_v(mdp_solver.get_optimal_v()),
               filename="v_{}.png".format(agent.episode))

def run_objective(space):
    aux_agent_configs = {"num_hidden_layers": FLAGS.num_hidden_layers,
                         "num_units": FLAGS.num_units,
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
        total_rmsve, avg_steps, values, errors = experiment.run_chain(
            agent=agent,
            environment=env,
            model_class=space["env_config"]["model_class"],
            num_episodes=space["env_config"]["num_episodes"],
            plot_curves=space["plot_curves"],
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            log_period=space["log_period"],
        )
    else:
        total_rmsve, avg_steps, values, errors = experiment.run_episodic(
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
    return total_rmsve, avg_steps, values, errors, env, agent, mdp_solver

if __name__ == '__main__':
    app.run(main)
