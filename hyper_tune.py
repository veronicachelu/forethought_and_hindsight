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
flags.DEFINE_string('env', 'random',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
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
    best_aoc_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_aoc_hyperparams.csv".format(FLAGS.env))
    best_min_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_min_hyperparams.csv".format(FLAGS.env))
    best_start_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_start_hyperparams.csv".format(FLAGS.env))

    persistent_agent_config = configs.agent_config.config[FLAGS.agent]
    env_config, volatile_agent_config = load_env_and_volatile_configs(FLAGS.env)

    interm_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    interm_fieldnames.extend(["seed", "steps", 'rmsve_aoc', 'rmsve_min', 'rmsve_start'])

    final_fieldnames = list(volatile_agent_config[FLAGS.agent].keys())
    final_fieldnames.extend(["steps", 'rmsve_aoc', 'rmsve_min', 'rmsve_start'])

    best_fieldnames = ["agent"]
    best_fieldnames.extend(list(volatile_agent_config[FLAGS.agent].keys()))
    best_fieldnames.extend(["steps", 'rmsve_aoc', 'rmsve_min', 'rmsve_start'])

    files = [interm_hyperparam_file, final_hyperparam_file,
                 best_aoc_hyperparam_file, best_min_hyperparam_file,
                 best_start_hyperparam_file]
    fieldnames = [interm_fieldnames, final_fieldnames, best_fieldnames,
                  best_fieldnames, best_fieldnames]
    for file, fieldname in zip(files, fieldnames):
        if not os.path.exists(file):
            with open(file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldname)
                writer.writeheader()

    limited_volatile_to_run, volatile_to_run = build_hyper_list(FLAGS.agent,
                                                                volatile_agent_config)

    for planning_depth, replay_capacity, lr, lr_p, lr_m in volatile_to_run:
        seed_config = {"planning_depth": planning_depth,
                      "replay_capacity": replay_capacity,
                      "lr": lr,
                      "lr_m": lr_m,
                      "lr_p": lr_p}
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

                total_rmsve, final_rmsve, start_rmsve, avg_steps = run_objective(space)

                with open(interm_hyperparam_file, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                    seed_config["rmsve_aoc"] = round(total_rmsve, 2)
                    seed_config["rmsve_min"] = round(final_rmsve, 2)
                    seed_config["rmsve_start"] = round(start_rmsve, 2)
                    seed_config["steps"] = avg_steps
                    writer.writerow(seed_config)

        if not configuration_exists(final_hyperparam_file, final_config, final_attributes):
            rmsve_aoc_avg, rmsve_min_avg, rmsve_start_avg, steps_avg = \
                get_avg_over_seeds(interm_hyperparam_file, final_config, final_attributes)
            with open(final_hyperparam_file, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                final_config["rmsve_aoc"] = round(rmsve_aoc_avg, 2)
                final_config["rmsve_min"] = round(rmsve_min_avg, 2)
                final_config["rmsve_start"] = round(rmsve_start_avg, 2)
                final_config["steps"] = steps_avg
                writer.writerow(final_config)

    for planning_depth, replay_capacity in limited_volatile_to_run:
        best_config = {"agent": FLAGS.agent,
                       "planning_depth": planning_depth,
                       "replay_capacity": replay_capacity, }
        best_attributes = list(best_config.keys())

        files = [best_aoc_hyperparam_file, best_min_hyperparam_file,
                 best_start_hyperparam_file]
        objectives = ["aoc", "min", "start"]
        for file, obj in zip(files, objectives):
            if not configuration_exists(file, best_config, best_attributes):
                the_best_hyperparms = get_best_over_final(final_hyperparam_file,
                                                          best_config, best_attributes[1:],
                                                          obj)
                with open(file, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=best_fieldnames)
                    for key, value in the_best_hyperparms.items():
                        best_config[key] = value
                    writer.writerow(best_config)

def get_avg_over_seeds(interm_hyperparam_file, final_config, final_attributes):
    rmsve_aoc_avg = []
    rmsve_min_avg = []
    rmsve_start_avg = []
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
                rmsve_aoc_avg.append(float(row['rmsve_aoc']))
                rmsve_min_avg.append(float(row['rmsve_min']))
                rmsve_start_avg.append(float(row['rmsve_start']))
                steps_avg.append(float(row['steps']))
        return np.mean(rmsve_aoc_avg), np.mean(rmsve_min_avg), np.mean(rmsve_start_avg),\
               np.mean(steps_avg, dtype=int)

def get_best_over_final(final_hyperparam_file, best_config, best_attributes, objective):
    rmsve = np.infty
    objective2key = {"aoc": 'rmsve_aoc', "min": 'rmsve_min', "start": 'rmsve_start'}
    with open(final_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in best_attributes:
                if float(row[key]) != float(best_config[key]):
                    ok = False
                    break
            if ok == True:
                if float(row[objective2key[objective]]) < rmsve:
                    best_config = row
                    rmsve = float(row[objective2key[objective]])
        return best_config

def configuration_exists(hyperparam_file, crt_config, attributes):
    with open(hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in attributes:
                if str(row[key]) != str(crt_config[key]):
                    ok = False
                    break
            if ok == True:
                return True
        return False

def run_objective(space):
    aux_agent_configs = {
                         "batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len}

    seed = space["crt_config"]["seed"]
    if space["env_config"]["non_gridworld"]:
        env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)
        total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_chain(
            agent=agent,
            model_class=space["env_config"]["model_class"],
            mdp_solver=mdp_solver,
            environment=env,
            num_episodes=space["env_config"]["num_episodes"],
            plot_curves=space["plot_curves"],
            plot_errors=space["plot_errors"],
            plot_values=space["plot_values"],
            log_period=space["log_period"],
        )
    else:
        env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)
        total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_episodic(
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
    return total_rmsve, final_rmsve, start_rmsve, avg_steps

if __name__ == '__main__':
    app.run(main)
