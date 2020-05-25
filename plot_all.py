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
# style.use('ggplot')
style.use("default")
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
# flags.DEFINE_string('env', "bipartite_100_1", 'where to save results')
flags.DEFINE_string('env', "bipartite_", 'where to save results')
# flags.DEFINE_string('env', "fanin", 'where to save results')
flags.DEFINE_bool('tabular', False, 'where to save results')
flags.DEFINE_bool('mle', False, 'where to save results')
flags.DEFINE_bool('mb', True, 'where to save results')
# flags.DEFINE_bool('mb', False, 'where to save results')
# flags.DEFINE_bool('paml', True, 'where to save results')
flags.DEFINE_bool('paml', False, 'where to save results')
flags.DEFINE_string('pivoting', "bw_c_fw_p", 'where to save results')
# flags.DEFINE_string('pivoting', "corr", 'where to save results')
# flags.DEFINE_string('pivoting', "MLE_PAML", 'where to save results')
flags.DEFINE_float('lr', 0.1, 'where to save results')
# flags.DEFINE_string('env', "random_linear", 'where to save results')
flags.DEFINE_float('ymin', None, 'plot up to')
# flags.DEFINE_float('ymin', 0.65, 'plot up to')
flags.DEFINE_float('ymax', None, 'plot up to')
# flags.DEFINE_float('ymax', 1.75, 'plot up to')
flags.DEFINE_bool('cumulative_rmsve', False, 'n-step plot or comparison plt')
# flags.DEFINE_bool('cumulative_rmsve', True, 'n-step plot or comparison plt')
# flags.DEFINE_integer('ˀnum_runs', 100, '')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
FLAGS = flags.FLAGS
FONTSIZE = 17
LINEWIDTH = 2

mle_paml_dashed = {
    "p_fw_PAML": "p_fw_MLE",
    "p_bw_PAML": "p_bw_MLE",
          # "p_fw_proj_PAML": "p_fw_PAML",
          # "p_bw_proj_PAML": "p_bw_PAML",
          # "p_bw_proj_MLE": "p_bw_MLE",
          # "p_fw_proj_MLE": "p_fw_MLE",
}

mle_paml_dotted = {
    "p_fw_proj_PAML": "p_fw_proj_MLE",
    "p_bw_proj_PAML": "p_bw_proj_MLE"
}

mle_paml_dash_dotted = {}
# mle_paml_dash_dotted = {
#     "p_bw_proj_MLE": "p_bw_MLE",
#     "p_fw_proj_MLE": "p_fw_MLE"
# }

mle_dashed = {
          "p_true_bw_recur": "p_bw_recur_MLE",
          "c_true_bw_recur": "c_bw_recur_MLE",
          "p_true_bw": "p_bw_MLE",
          "c_true_bw": "c_bw_MLE",
          "p_true_fw": "p_fw_MLE",
          "c_true_fw": "c_fw_MLE",
          # "p_bw_PAML": "p_bw_MLE",
          # "c_bw_PAML": "c_bw_MLE",
          # "p_fw_PAML": "p_fw_MLE",
          # "c_fw_PAML": "c_fw_MLE",
          }
mb_dashed = {
          "mb_c_true_bw": "mb_c_bw",
          "mb_p_true_bw": "mb_p_bw",
          "mb_p_true_fw": "mb_p_fw",
          "mb_c_true_fw": "mb_c_fw",
          "mb_p_true_bw_recur": "mb_p_bw_recur",
          "mb_c_true_bw_recur": "mb_c_bw_recur",
          }
mb_mle_dashed = {
          "mb_p_true_bw": "mb_p_bw_MLE",
          "mb_c_true_bw": "mb_c_bw_MLE",
          "mb_c_true_bwfw": "mb_c_bwfw_MLE",
          "mb_p_true_fw": "mb_p_fw_MLE",
          "mb_c_true_fw": "mb_c_fw_MLE",
          "mb_p_true_bw_recur": "mb_p_bw_recur_MLE",
          "mb_c_true_bw_recur": "mb_c_bw_recur_MLE",
}

dashed = {
          "p_bw_PAML": "p_bw",
          "c_bw_PAML": "c_bw",
          "p_fw_PAML": "p_fw",
          "c_fw_PAML": "c_fw",

          # "p_bw_PAML_MLE": "p_bw",
          # "c_bw_PAML_MLE": "c_bw",
          # "p_fw_PAML_MLE": "p_fw",
          # "c_fw_PAML_MLE": "c_fw",
          # "p_true_bw_recur": "p_bw_recur",
          # "c_true_bw_recur": "c_bw_recur",

          }

dotted = {
          "p_true_bw": "p_bw",
          "c_true_bw": "c_bw",
          "p_true_fw": "p_fw",
          "c_true_fw": "c_fw",
}

# dotted = ["true_bw", "true_fw", "mb_true_fw", "mb_true_bw",
#           "true_bw_recur", "mb_true_bw_recur"]

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    logs = os.path.join(best_hyperparam_folder, FLAGS.env)
    plots = os.path.join(FLAGS.plots, FLAGS.env)

    if not os.path.exists(plots):
        os.makedirs(plots)

    env_config, volatile_agent_config = load_env_and_volatile_configs(FLAGS.env)

    name = FLAGS.pivoting + "_" +"all"
    if FLAGS.mb:
        name = "mb_" + name
    if FLAGS.mle:
        name = name + "_mle"

    internal_dashed = dashed
    internal_dotted = dotted
    internal_dash_dotted = {}
    if FLAGS.mle and FLAGS.mb:
        internal_dashed = mb_mle_dashed
    elif FLAGS.mle and FLAGS.paml:
        internal_dashed = mle_paml_dashed
        internal_dotted = mle_paml_dotted
        internal_dash_dotted = mle_paml_dash_dotted
    elif FLAGS.mle:
        internal_dashed = mle_dashed
    elif FLAGS.mb:
        internal_dashed = mb_dashed

    comparison_config = configs.comparison_configs.configs[FLAGS.env][name]

    unique_color_configs = [c for c in comparison_config["agents"]
                            if c not in internal_dashed.keys()]
    n = len(unique_color_configs)

    colors = ["C{}".format(c) for c in range(n)]
    alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}

    persistent_agent_config = configs.agent_config.config["vanilla"]
    plot_for_agent("vanilla", env_config, persistent_agent_config,
                   volatile_agent_config, 0, 0, logs, "gray", "-")

    for i, agent in enumerate(comparison_config["agents"]):
        if agent not in internal_dashed.keys() and\
                        agent not in internal_dotted.keys() and \
                        agent not in internal_dash_dotted.keys():
            color = alg_to_color[agent]
            linestyle = "-"
        elif agent in internal_dashed.keys():
            color = alg_to_color[internal_dashed[agent]]
            linestyle = "--"
        elif agent in internal_dotted.keys():
            color = alg_to_color[internal_dotted[agent]]
            linestyle = ":"
        elif agent in internal_dash_dotted.keys():
            color = alg_to_color[internal_dash_dotted[agent]]
            linestyle = "-."

        planning_depth = comparison_config["planning_depths"][i]
        max_norm = None
        if FLAGS.paml and "max_norms" in comparison_config.keys():
            max_norm = comparison_config["max_norms"][i]
        replay_capacity = comparison_config["replay_capacities"][i]
        persistent_agent_config = configs.agent_config.config[agent]
        plot_for_agent(agent, env_config, persistent_agent_config,
                       volatile_agent_config, planning_depth,
                       replay_capacity, logs, color, linestyle, max_norm)


    if FLAGS.cumulative_rmsve:
        yaxis = 'Cumulative RMSVE'
        xaxis = "Timesteps"
    else:
        yaxis = 'RMSVE'
        xaxis = "Episodes"

    # Set the y limits
    if FLAGS.ymin is not None and FLAGS.ymax is not None:
        plt.ylim(FLAGS.ymin, FLAGS.ymax)
    plt.ylabel(yaxis, fontsize=FONTSIZE)
    plt.xlabel(xaxis, fontsize=FONTSIZE)
    plt.legend(
        # loc='lower right' if FLAGS.cumulative_rmsve else 'upper right',
               frameon=True, ncol = 2,  mode = "expand",
               loc = 'lower left',
               borderaxespad=0.,
               prop={'size': FONTSIZE}, bbox_to_anchor = (0., 1.02, 1., .102))
               # bbox_to_anchor=(1.1, 1.1))
    if not os.path.exists(plots):
        os.makedirs(plots)

    name = FLAGS.pivoting + "_" +"all"
    if FLAGS.mb:
        name = "mb_" + name
    if FLAGS.mle:
        name = name + "_mle"

    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots,
                             "{}_{}.png".format(name,
                                                "CumRMSVE" if
                                                FLAGS.cumulative_rmsve else
                                                "RMSVE")))

def plot_for_agent(agent, env_config, persistent_agent_config,
                   volatile_agent_config, planning_depth, replay_capacity, logs, color, linestyle, max_norm=None):
    print(agent)
    # if not FLAGS.tabular:
    if max_norm is None or max_norm == 0:
        log_folder_agent = os.path.join(logs, "{}_{}_{}".format(persistent_agent_config["run_mode"], planning_depth, replay_capacity))
    else:
        log_folder_agent = os.path.join(logs, "{}_{}_{}_{}".format(persistent_agent_config["run_mode"], planning_depth,
                                                                replay_capacity, max_norm))
    # else:
    #     log_folder_agent = os.path.join(logs, "{}_{}_{}_{}".format(persistent_agent_config["run_mode"], planning_depth,
    #                                                             replay_capacity, FLAGS.lr))
    volatile_config = {"agent": agent,
                       "planning_depth": planning_depth,
                       "max_norm": max_norm,
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
        #print("seed_{}_agent_{}".format(seed, space["crt_config"]["agent"]))
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
            print("no tags")
            continue

        msve = event_acc.Tensors(tag)

        x = [m[1] for m in msve]
        y = [tf.make_ndarray(m[2]) for m in msve]
        all_y_over_seeds.append(np.array(y))

    # those_that_are_not_99 = [i for i, a in enumerate(all_y_over_seeds) if len(a) != 199]
    # print(those_that_are_not_99)
    #print(len(all_y_over_seeds))
    # all_y_over_seeds = [a[:99] for a in all_y_over_seeds]
    first_seed_size = len(all_y_over_seeds[0])
    the_incomplete = [i for i, a in enumerate(all_y_over_seeds) if len(a) != first_seed_size]
    print(the_incomplete)
    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    if space["crt_config"]["agent"] == "vanilla":
        plt.plot(x, mean_y_over_seeds, label="model-free", c="gray", alpha=1, linewidth=LINEWIDTH, linestyle="-")
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         color="gray", alpha=0.07)
    else:
        label = space["crt_config"]["agent"]
        if FLAGS.paml and space["crt_config"]["max_norm"] is not None:
            label += "_{}".format(space["crt_config"]["max_norm"])
        plt.plot(x, mean_y_over_seeds, label=label,
                 alpha=1, linewidth=LINEWIDTH, color=color,
                 linestyle=linestyle)
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         alpha=0.07, color=color,
                         linestyle=linestyle)

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
