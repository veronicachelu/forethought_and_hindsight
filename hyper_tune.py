from absl import app
from absl import flags
from jax import random as jrandom
from tqdm import tqdm

import agents
import experiment
import network
import utils
from utils import *
import csv
import copy
import configs
from main_utils import *

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
flags.DEFINE_integer('planning_depth', 1, 'Planning depth for MCTS')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .95, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
FLAGS = flags.FLAGS

def main(argv):
    all_hyperparam_folder = os.path.join(os.path.join(FLAGS.logs, "hyper"))
    env_hyperparam_folder = os.path.join(all_hyperparam_folder, FLAGS.env)
    agent_env_hyperparam_folder = os.path.join(env_hyperparam_folder, FLAGS.agent)
    if not os.path.exists(agent_env_hyperparam_folder):
        os.makedirs(agent_env_hyperparam_folder)
    interm_hyperparam_file = os.path.join(agent_env_hyperparam_folder, "interm_hyperparams.csv")
    final_hyperparam_file = os.path.join(agent_env_hyperparam_folder, "final_hyperparams.csv")
    best_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_hyperparams.csv".format(FLAGS.env))

    persistent_agent_config = configs.agent_config.config[FLAGS.agent]
    env_config, volatile_agent_config = load_env_and_volatile_configs(FLAGS.env)

    interm_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    interm_fieldnames.extend(["seed", "steps", 'rmsve'])

    final_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    final_fieldnames.extend(["steps", 'rmsve'])

    best_fieldnames = ["agent"]
    best_fieldnames.extend(list(volatile_agent_config[FLAGS.agent].keys()))
    best_fieldnames.extend(["steps", 'rmsve'])

    if not os.path.exists(interm_hyperparam_file):
        with open(interm_hyperparam_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
            writer.writeheader()
    if not os.path.exists(final_hyperparam_file):
        with open(final_hyperparam_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames)
            writer.writeheader()
    if not os.path.exists(best_hyperparam_file):
        with open(best_hyperparam_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=best_fieldnames)
            writer.writeheader()

    limited_volatile_to_run, volatile_to_run = build_hyper_list(volatile_agent_config)

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
            if not configuration_exists(interm_hyperparam_file,
                                        seed_config, attributes):
                space = {
                    "logs": None,
                    "plot_errors": False,
                    "plot_values": False,
                    "plot_curves": False,
                    "log_period": FLAGS.log_period,
                    "env_config": env_config,
                    "agent_config": persistent_agent_config,
                    "crt_config": seed_config}

                total_rmsve, avg_steps = run_objective(space)

                with open(interm_hyperparam_file, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                    seed_config["rmsve"] = round(total_rmsve, 2)
                    seed_config["steps"] = avg_steps
                    writer.writerow(seed_config)

        if not configuration_exists(final_hyperparam_file, final_config, final_attributes):
            rmsve_avg, steps_avg = get_avg_over_seeds(interm_hyperparam_file,
                                                      final_config, final_attributes)
            with open(final_hyperparam_file, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                final_config["rmsve"] = round(rmsve_avg, 2)
                final_config["steps"] = steps_avg
                writer.writerow(final_config)

    for planning_depth, replay_capacity in limited_volatile_to_run:
        best_config = {"agent": FLAGS.agent,
                       "planning_depth": planning_depth,
                       "replay_capacity": replay_capacity, }
        best_attributes = list(best_config.keys())

        if not configuration_exists(best_hyperparam_file, best_config, best_attributes):
            the_best_hyperparms = get_best_over_final(final_hyperparam_file,
                                                      best_config, best_attributes[1:])
            with open(best_hyperparam_file, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=best_fieldnames)
                for key, value in the_best_hyperparms.items():
                    best_config[key] = value
                writer.writerow(best_config)

def build_hyper_list(volatile_agent_config):
    volatile_to_run = []
    limited_volatile_to_run = []
    for planning_depth in volatile_agent_config[FLAGS.agent]["planning_depth"]:
        for replay_capacity in volatile_agent_config[FLAGS.agent]["replay_capacity"]:
            limited_volatile_to_run.append([planning_depth, replay_capacity])
            for lr in volatile_agent_config[FLAGS.agent]["lr"]:
                for lr_m in volatile_agent_config[FLAGS.agent]["lr_m"]:
                    volatile_to_run.append([planning_depth, replay_capacity,
                                            round(lr, 2), round(lr_m, 2)])
    return limited_volatile_to_run, volatile_to_run

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
                rmsve_avg.append(float(row['rmsve']))
                steps_avg.append(float(row['steps']))
        return np.mean(rmsve_avg), np.mean(steps_avg, dtype=int)

def get_best_over_final(final_hyperparam_file, best_config, best_attributes):
    rmsve = np.infty
    with open(final_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in best_attributes:
                if float(row[key]) != float(best_config[key]):
                    ok = False
                    break
            if ok == True:
                if float(row['rmsve']) < rmsve:
                    best_config = row
                    rmsve = float(row['rmsve'])
        return best_config

def configuration_exists(hyperparam_file, crt_config, attributes):
    with open(hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in attributes:
                if key != crt_config[key] and float(row[key]) != float(crt_config[key]):
                    ok = False
                    break
            if ok == True:
                return True
        return False

def run_objective(space):
    aux_agent_configs = {"num_hidden_layers": FLAGS.num_hidden_layers,
                         "num_units": FLAGS.num_units,
                         "batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len}

    seed = space["crt_config"]["seed"]
    if space["env_config"]["non_gridworld"]:
        env, agent, _ = run_experiment(seed, space, aux_agent_configs)
        total_rmsve, avg_steps = experiment.run_chain(
            agent=agent,
            model_class=space["env_config"]["model_class"],
            environment=env,
            num_episodes=space["env_config"]["num_episodes"],
            plot_curves=space["plot_curves"],
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            log_period=space["log_period"],
        )
        return total_rmsve, avg_steps
    else:
        env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)
        total_rmsve, avg_steps = experiment.run_episodic(
            agent=agent,
            environment=env,
            mdp_solver=mdp_solver,
            model_class=space["env_config"]["model_class"],
            num_episodes=space["env_config"]["num_episodes"],
            max_len=FLAGS.max_len,
            plot_curves=space["plot_curves"],
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            log_period=space["log_period"],
        )
        return total_rmsve, avg_steps

if __name__ == '__main__':
    app.run(main)
