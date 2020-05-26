

import csv
from copy import deepcopy
import configs
from main_utils import *

flags.DEFINE_string('agent', 'q', 'what agent to run')
flags.DEFINE_string('env', 'maze_05', 'env')
flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_integer('log_period', 1, 'Log summaries every .... episodes.')
flags.DEFINE_integer('max_len', 400, 'Maximum number of time steps an episode may last (default: 100).')
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

def build_hyper_list(agent, volatile_agent_config, up_to=None):
    volatile_to_run = []
    limited_volatile_to_run = []
    for planning_depth in volatile_agent_config[agent]["planning_depth"]:
        if up_to is not None and planning_depth > up_to:
            break
        for replay_capacity in volatile_agent_config[agent]["replay_capacity"]:
            limited_volatile_to_run.append([planning_depth, replay_capacity])
            for lr_ctrl in volatile_agent_config[agent]["lr_ctrl"]:
                for lr_m in volatile_agent_config[agent]["lr_m"]:
                    volatile_to_run.append([planning_depth, replay_capacity,
                                            round(lr_ctrl, 6), round(lr_m, 6)])
    return limited_volatile_to_run, volatile_to_run

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
    interm_fieldnames.extend(["seed", "reward"])

    final_fieldnames = list(volatile_agent_config[agent].keys())
    final_fieldnames.extend(["reward"])

    best_fieldnames = ["agent"]
    best_fieldnames.extend(list(volatile_agent_config[agent].keys()))

    best_fieldnames.extend(['reward'])

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


    for planning_depth, replay_capacity, lr_ctrl, lr_m in volatile_to_run:
        if agent != "vanilla":
            lr_ctrl = lr_vanilla
        seed_config = {"planning_depth": planning_depth,
                      "replay_capacity": replay_capacity,
                      "lr_ctrol": lr_ctrl,
                      "lr_m": lr_m,
                      }
        final_config = deepcopy(seed_config)
        attributes = list(seed_config.keys())
        attributes.append("seed")

        final_attributes = list(final_config.keys())

        # for seed in tqdm(range(0, env_config["num_runs"])):
        for seed in range(0, env_config["num_runs"]):
            seed_config["seed"] = seed
            if not configuration_exists(interm_hyperparam_file,
                                        seed_config, attributes):
                space = {
                    "logs": env_hyperparam_folder,
                    "plot_errors": False,
                    "plot_values": False,
                    "plot_curves": True,
                    "log_period": FLAGS.log_period,
                    "env_config": env_config,
                    "agent_config": persistent_agent_config,
                    "crt_config": seed_config}

                reward, _ = run_objective(space)
                seed_config.pop("lr_ctrl")
                with open(interm_hyperparam_file, 'a+', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=interm_fieldnames)
                    seed_config["reward"] = round(reward, 2)
                    writer.writerow(seed_config)

        if not configuration_exists(final_hyperparam_file, final_config, final_attributes):
            reward, reward_std = \
                get_avg_over_seeds(interm_hyperparam_file, final_config, final_attributes)
            with open(final_hyperparam_file, 'a+', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=final_fieldnames)
                final_config["reward"] = round(reward, 2)
                final_config["reward_std"] = round(reward_std, 2)
                writer.writerow(final_config)

    for planning_depth, replay_capacity in limited_volatile_to_run:
        best_config = {"agent": agent,
                       "planning_depth": planning_depth,
                       "replay_capacity": replay_capacity, }
        best_attributes = list(best_config.keys())

        files = [best_aoc_hyperparam_file, best_min_hyperparam_file,
                 best_start_hyperparam_file]
        objectives = ["reward"]
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
    reward_avg = []
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
                reward_avg.append(float(row['reward']))
        return np.mean(reward_avg), np.std(reward_avg)

def get_best_over_final(final_hyperparam_file, best_config, best_attributes, objective):
    rmsve_over_std = np.infty
    objective2key = {"reward": 'reward'}
    with open(final_hyperparam_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok = True
            for key in best_attributes:
                if float(row[key]) != float(best_config[key]):
                    ok = False
                    break
            if ok == True:
                if float(row[objective2key[objective]]) + \
                    float(row[objective2key[objective] + "_std"]) < rmsve_over_std:
                    best_config = row
                    rmsve_over_std = float(row[objective2key[objective]]) + \
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

    aux_agent_configs["mb"] = True if space["agent_config"]["run_mode"].split("_")[0] == "mb" else False
    if space["agent_config"]["run_mode"].split("_")[0] == "mb":
        aux_agent_configs["pivot"] = space["agent_config"]["run_mode"].split("_")[1]
    else:
        aux_agent_configs["pivot"] = space["agent_config"]["run_mode"].split("_")[0]

    total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors = experiment.run_episodic(
        agent=agent,
        space=space,
        aux_agent_configs=aux_agent_configs,
        mdp_solver=mdp_solver,
        environment=env,
    )
    # print(total_rmsve, final_rmsve, start_rmsve, avg_steps, values, errors)
    return total_rmsve, final_rmsve, start_rmsve, avg_steps

if __name__ == '__main__':
    app.run(main)
