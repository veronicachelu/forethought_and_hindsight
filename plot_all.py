import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from absl import app
from absl import flags
import matplotlib.style as style
import cycler
from main_utils import *
import glob
style.available
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
# style.use("classic")
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_string('env', "linear_maze", 'where to save results')
# flags.DEFINE_bool('cumulative_rmsve', False, 'n-step plot or comparison plt')
flags.DEFINE_bool('cumulative_rmsve', True, 'n-step plot or comparison plt')
# flags.DEFINE_integer('num_runs', 100, '')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
FLAGS = flags.FLAGS
FONTSIZE = 30
LINEWIDTH = 4

dashed = {"bw_fw": "bw", "fw_pri": "fw_rnd", "bw_fw_MG": "bw_fw_PWMA"}

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)
    plots = os.path.join(FLAGS.plots, FLAGS.env)

    if not os.path.exists(plots):
        os.makedirs(plots)

    env_config, volatile_agent_config = load_env_and_volatile_configs(FLAGS.env)

    comparison_config = configs.comparison_configs.configs[FLAGS.env]["all"]

    unique_color_configs = [c for c in comparison_config["agents"] if c not in dashed.keys()]
    n = len(unique_color_configs)

    cmap_color = plt.cm.winter(np.linspace(0.0, 1.0, n)[::-1])
    # hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
    #                tuple(color[:, 0:-1]))
    # color = hexcolor  # plt.cm.viridis(np.linspace(0, 1, n))
    # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
    colors = ['#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)) for rgb in
                   tuple(cmap_color[:, 0:-1])]
    alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}
    for i, agent in enumerate(comparison_config["agents"]):
        if agent not in dashed:
            color = alg_to_color[agent]
            linestyle = "-"
        else:
            color = alg_to_color[dashed[agent]]
            linestyle = "-."
        planning_depth = comparison_config["planning_depths"][i]
        replay_capacity = comparison_config["replay_capacities"][i]
        persistent_agent_config = configs.agent_config.config[agent]
        plot_for_agent(agent, env_config, persistent_agent_config,
                       volatile_agent_config, planning_depth, replay_capacity, logs, color, linestyle)

    persistent_agent_config = configs.agent_config.config["vanilla"]
    plot_for_agent("vanilla", env_config, persistent_agent_config,
                   volatile_agent_config, 0, 0, logs, "r", ":")


    if FLAGS.cumulative_rmsve:
        yaxis = 'Cumulative RMSVE'
        xaxis = "Timesteps"
    else:
        yaxis = 'RMSVE'
        xaxis = "Episodes"

    plt.ylabel(yaxis, fontsize=FONTSIZE)
    plt.xlabel(xaxis, fontsize=FONTSIZE)
    plt.legend(loc='lower right' if FLAGS.cumulative_rmsve else 'upper right',
               frameon=True,
               prop={'size': FONTSIZE})
    if not os.path.exists(plots):
        os.makedirs(plots)

    plt.savefig(os.path.join(plots,
                             "{}_{}.png".format("all",
                                                "CumRMSVE" if
                                                FLAGS.cumulative_rmsve else
                                                "RMSVE")))

def plot_for_agent(agent, env_config, persistent_agent_config,
                   volatile_agent_config, planning_depth, replay_capacity, logs, color, linestyle):
    print(agent)
    log_folder_agent = os.path.join(logs, "{}_{}_{}".format(persistent_agent_config["run_mode"], planning_depth, replay_capacity))
    volatile_config = {"agent": agent,
                       "planning_depth": planning_depth,
                       "replay_capacity": replay_capacity,
                       "logs": log_folder_agent}
    space = {
    "env_config": env_config,
    "agent_config": persistent_agent_config,
    "crt_config": volatile_config}
    plot_tensorflow_log(space, color, linestyle)

def plot_tensorflow_log(space, color, linestyle):
    tf_size_guidance = {
        'compressedHistograms': 100000,
        'images': 0,
        'scalars': 200000,
        'histograms': 1,
        'tensors': 200000,
    }
    all_y_over_seeds = []
    num_runs = space["env_config"]["num_runs"]
    for seed in range(num_runs):
        # print("seed_{}_agent_{}".format(seed, space["crt_config"]["agent"]))
        logs = os.path.join(os.path.join(space["crt_config"]["logs"],
                                         "summaries"),
                                        "seed_{}".format(seed))
        list_of_files = glob.glob(os.path.join(logs, '*'))  # * means all if need specific format then *.csv
        if len(list_of_files) == 0:
            print("no files in folder {}".format(logs))
            return
        if len(list_of_files) > 1:
            print("ERROR, there should be only one file in folder {}".format(logs))
        filename = list_of_files[0]
        filepath = os.path.join(logs, filename)
        event_acc = EventAccumulator(filepath, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        # print(event_acc.Tags())
        if FLAGS.cumulative_rmsve:
            tag = 'train/total_rmsve'
        else:
            tag = 'train/rmsve'
        if not tag in event_acc.Tags()["tensors"]:
            return

        msve = event_acc.Tensors(tag)

        x = [m[1] for m in msve]
        y = [tf.make_ndarray(m[2]) for m in msve]
        all_y_over_seeds.append(np.array(y))

    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    if space["crt_config"]["agent"] == "vanilla":
        plt.plot(x, mean_y_over_seeds, label="vanilla", c="r", alpha=1, linewidth=LINEWIDTH, linestyle=":")
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         color="r", alpha=0.2)
    else:
        plt.plot(x, mean_y_over_seeds, label=format_name(space["crt_config"]["agent"],
                                            space["crt_config"]["planning_depth"],
                                            space["crt_config"]["replay_capacity"]),
                 alpha=1, linewidth=LINEWIDTH, color=color,
                 linestyle=linestyle)
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         alpha=0.2)

def format_name(agent, planning_perf, replay_capacity):
    if not(planning_perf == 0):
        if not (replay_capacity == 0):
            return "{}_{}_{}".format(agent, planning_perf, replay_capacity)
        else:
            return "{}_{}".format(agent, planning_perf)
    else:
        if not (replay_capacity == 0):
            return "{}_{}".format(agent, replay_capacity)
        else:
            return "{}".format(agent)

if __name__ == '__main__':
    app.run(main)
