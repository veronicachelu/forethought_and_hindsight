

import csv
from copy import deepcopy
import configs
from main_utils import *

flags.DEFINE_string('agent', 'bw', 'what agent to run')
flags.DEFINE_string('env', 'split', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 100000, 'Maximum number of time steps an episode may last (default: 100).')
flags.DEFINE_integer('num_hidden_layers', 0, 'number of hidden layers')
flags.DEFINE_integer('planning_iter', 1, 'Number of minibatches of model-based backups to run for planning')
flags.DEFINE_integer('planning_period', 1, 'Number of timesteps of real experience to see before running planning')
flags.DEFINE_integer('model_learning_period', 1,
                     'Number of steps timesteps of real experience to cache before updating the model')
flags.DEFINE_integer('batch_size', 1, 'size of batches sampled from replay')
flags.DEFINE_float('discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('min_replay_size', 1, 'min replay size before training.')
FLAGS = flags.FLAGS

def main(argv):
    all_hyperparam_folder = os.path.join(os.path.join(FLAGS.logs, "hyper"))
    env_hyperparam_folder = os.path.join(all_hyperparam_folder, FLAGS.env)
    agent_env_hyperparam_folder = os.path.join(env_hyperparam_folder, FLAGS.agent)
    if not os.path.exists(agent_env_hyperparam_folder):
        os.makedirs(agent_env_hyperparam_folder)

    best_aoc_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_aoc_hyperparams.csv".format(FLAGS.env))

    lr_vanilla = None
    if FLAGS.agent != "vanilla":
        lr_vanilla = get_lr(best_aoc_hyperparam_file, "vanilla")
        if lr_vanilla == None:
            run_for_agent("vanilla")
            lr_vanilla = get_lr(best_aoc_hyperparam_file, "vanilla")

    run_for_agent(FLAGS.agent, lr_vanilla)

def run_for_agent(agent, lr_vanilla=None):
    all_hyperparam_folder = os.path.join(os.path.join(FLAGS.logs, "hyper"))
    env_hyperparam_folder = os.path.join(all_hyperparam_folder, FLAGS.env)
    agent_env_hyperparam_folder = os.path.join(env_hyperparam_folder, agent)
    if not os.path.exists(agent_env_hyperparam_folder):
        os.makedirs(agent_env_hyperparam_folder)

    best_aoc_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_aoc_hyperparams.csv".format(FLAGS.env))

    interm_hyperparam_file = os.path.join(agent_env_hyperparam_folder, "interm_hyperparams.csv")
    final_hyperparam_file = os.path.join(agent_env_hyperparam_folder, "final_hyperparams.csv")
    best_min_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_min_hyperparams.csv".format(FLAGS.env))
    best_start_hyperparam_file = os.path.join(env_hyperparam_folder, "{}_best_start_hyperparams.csv".format(FLAGS.env))

    persistent_agent_config = configs.agent_config.config[agent]
    env_config, volatile_agent_config = load_env_and_volatile_configs(FLAGS.env)


    interm_fieldnames = list(volatile_agent_config[agent].keys())
    interm_fieldnames.extend(["seed", "steps", 'rmsve_aoc', 'rmsve_min', 'rmsve_start'])

    final_fieldnames = list(volatile_agent_config[agent].keys())
    final_fieldnames.extend(["steps", 'rmsve_aoc', 'rmsve_aoc_std',
                             'rmsve_min', 'rmsve_min_std',
                             'rmsve_start', 'rmsve_start_std'])

    best_fieldnames = ["agent"]
    best_fieldnames.extend(list(volatile_agent_config[agent].keys()))
    best_fieldnames.remove("lr_ctrl")
    final_fieldnames.remove("lr_ctrl")
    interm_fieldnames.remove("lr_ctrl")

    best_fieldnames.extend(["steps", 'rmsve_aoc', 'rmsve_aoc_std',
                            'rmsve_min', 'rmsve_min_std',
                            'rmsve_start', 'rmsve_start_std'])

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

    limited_volatile_to_run, volatile_to_run = build_hyper_list(agent,
                                                                volatile_agent_config)


    for planning_depth, replay_capacity, lr, lr_p, lr_m, lr_ctrl in volatile_to_run:
        if agent != "vanilla":
            lr = lr_vanilla
            lr_p = lr_vanilla
        seed_config = {"planning_depth": planning_depth,
                      "replay_capacity": replay_capacity,
                      "lr": lr,
                      "lr_m": lr_m,
                      "lr_p": lr_p}
        final_config = deepcopy(seed_config)
        attributes = list(seed_config.keys())
        attributes.append("seed")

        final_attributes = list(final_config.keys())

        # for seed in tqdm(range(0, env_config["num_runs"])):
        for seed in range(0, env_config["num_runs"]):
            seed_config["seed"] = seed
            seed_config["lr_ctrl"] = lr_ctrl
            if not configuration_exists(interm_hyperparam_file,
                                        seed_config, attributes):
                space = {
                    "logs": env_hyperparam_folder,
                    "plot_errors": True,
                    "plot_values": True,
                    "plot_curves": True,
                    "log_period": FLAGS.log_period,
                    "env_config": env_config,
                    "agent_config": persistent_agent_config,
                    "crt_config": seed_config}

                total_rmsve, final_rmsve, start_rmsve, avg_steps = run_objective(space)
                seed_config.pop("lr_ctrl")
                with open(interm_hyperparam_file, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                    seed_config["rmsve_aoc"] = round(total_rmsve, 2)
                    seed_config["rmsve_min"] = round(final_rmsve, 2)
                    seed_config["rmsve_start"] = round(start_rmsve, 2)
                    seed_config["steps"] = avg_steps
                    writer.writerow(seed_config)

        if not configuration_exists(final_hyperparam_file, final_config, final_attributes):
            (rmsve_aoc_avg, rmsve_aoc_std), (rmsve_min_avg, rmsve_min_std),\
            (rmsve_start_avg, rmsve_start_std), steps_avg = \
                get_avg_over_seeds(interm_hyperparam_file, final_config, final_attributes)
            with open(final_hyperparam_file, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                final_config["rmsve_aoc"] = round(rmsve_aoc_avg, 2)
                final_config["rmsve_aoc_std"] = round(rmsve_aoc_std, 2)
                final_config["rmsve_min"] = round(rmsve_min_avg, 2)
                final_config["rmsve_min_std"] = round(rmsve_min_std, 2)
                final_config["rmsve_start"] = round(rmsve_start_avg, 2)
                final_config["rmsve_start_std"] = round(rmsve_start_std, 2)
                final_config["steps"] = steps_avg
                writer.writerow(final_config)

    for planning_depth, replay_capacity in limited_volatile_to_run:
        best_config = {"agent": agent,
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
                    try:
                        writer.writerow(best_config)
                    except:
                        print("Error")

def get_lr(best_hyperparam_file, agent="vanilla"):
    lr = None
    if not os.path.exists(best_hyperparam_file):
        return lr

    with open(best_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["agent"]) == agent:
                lr = float(row["lr"])
                break
        return lr

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
                if row['rmsve_start'] is None:
                    rmsve_start_avg.append(0)
                else:
                    rmsve_start_avg.append(float(row['rmsve_start']))
                steps_avg.append(float(row['steps']))
        return (np.mean(rmsve_aoc_avg), np.std(rmsve_aoc_avg)), (np.mean(rmsve_min_avg), np.std(rmsve_min_avg)),\
               (np.mean(rmsve_start_avg), np.std(rmsve_start_avg)), \
               np.mean(steps_avg, dtype=int)

def get_best_over_final(final_hyperparam_file, best_config, best_attributes, objective):
    rmsve_over_std = np.infty
    objective2key = {"aoc": 'rmsve_aoc',
                     "min": 'rmsve_min',
                     "start": 'rmsve_start'}
    with open(final_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in best_attributes:
                if float(row[key]) != float(best_config[key]):
                    ok = False
                    break
            if ok == True:
                if float(row[objective2key[objective]]) * \
                    float(row[objective2key[objective] + "_std"]) < rmsve_over_std:
                    best_config = row
                    rmsve_over_std = float(row[objective2key[objective]]) * \
                                     float(row[objective2key[objective] + "_std"])
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
    aux_agent_configs = {"batch_size": FLAGS.batch_size,
                         "discount": FLAGS.discount,
                         "min_replay_size": FLAGS.min_replay_size,
                         "model_learning_period": FLAGS.model_learning_period,
                         "planning_period": FLAGS.planning_period,
                         "max_len": FLAGS.max_len,
                         "log_period": FLAGS.log_period}
    seed = space["crt_config"]["seed"]
    env, agent, mdp_solver = run_experiment(seed, space, aux_agent_configs)
    total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_episodic(
        agent=agent,
        space=space,
        aux_agent_configs=aux_agent_configs,
        mdp_solver=mdp_solver,
        environment=env,
    )
    print(total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors)
    return total_rmsve, final_rmsve, start_rmsve, avg_steps

if __name__ == '__main__':
    app.run(main)
