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
import sklearn.metrics
style.available
style.use('seaborn-poster') #sets the size of the charts
# style.use('ggplot')
style.use("default")
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
# flags.DEFINE_string('env', "bipartite_100_1", 'where to save results')
flags.DEFINE_string('plotting', "mb+mfmb", 'where to save results')
# flags.DEFINE_string('config', "mfmb", 'where to save results')
flags.DEFINE_bool('tabular', True, 'where to save results')
flags.DEFINE_float('ymin', None, 'plot up to')
flags.DEFINE_float('ymax', None, 'plot up to')
flags.DEFINE_bool('cumulative_rmsve', False, 'n-step plot or comparison plt')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
FLAGS = flags.FLAGS
FONTSIZE = 23
TICKSIZE = 15
LINEWIDTH = 3
MARKERSIZE = 10

plottings = {
    "mb+mfmb": ["mb", "mfmb"]
}
plot_configs = {
    "mb": {
        "title": "Pure planning",
        "mb": True,
        "mle": True,
        "results": [
            {
                "env": "bipartite_100_1",
                "pivoting": "bw_c_fw_p",
                "mb": True,
                "mle": True,

            },
            {
                "env": "bipartite_10_1",
                "pivoting": "bw_c_fw_p",
                "mb": True,
                "mle": True,
            },
            {
                "env": "bipartite",
                "pivoting": "bw_c_fw_p",
                "mb": True,
                "mle": True,
            },
            {
                "env": "bipartite_1_10",
                "pivoting": "bw_c_fw_p",
                "mb": True,
                "mle": True,
            },
            {
                "env": "bipartite_1_100",
                "pivoting": "bw_c_fw_p",
                "mb": True,
                "mle": True,
            }
        ]
    },
    "mfmb": {
        "title": "Learning and planning",
        "mb": False,
        "mle": True,
        "results": [
            {
                "env": "bipartite_100_1",
                "pivoting": "bw_c_fw_p",
                "mb": False,
                "mle": True,

            },
            {
                "env": "bipartite_10_1",
                "pivoting": "bw_c_fw_p",
                 "mb": False,
                "mle": True,
            },
            {
                "env": "bipartite",
                "pivoting": "bw_c_fw_p",
                 "mb": False,
                "mle": True,
            },
            {
                "env": "bipartite_1_10",
                "pivoting": "bw_c_fw_p",
                 "mb": False,
                "mle": True,
            },
            {
                "env": "bipartite_1_100",
                "pivoting": "bw_c_fw_p",
                 "mb": False,
                "mle": True,
            }
        ]
    }
}
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

# naming = {
#     "vanilla": r"model_free",
#     "mb_p_bw_MLE": r"bw_plan($x$)",
#     "mb_c_bw_MLE": r"bw_plan($x^\prime$)",
#     "mb_p_fw_MLE": r"fw_plan($x$)",
#     "mb_c_fw_MLE": r"fw_plan($x^\prime$)",
#     "mb_p_true_fw": r"fw_plan($P^*;x$)",
#     "mb_c_true_fw": r"fw_plan($P^*;x^\prime$)",
#     "mb_p_true_bw": r"bw plan($\overleftarrow{P}^*;x$)",
#     "mb_c_true_bw": r"bw plan($\overleftarrow{P}^*;x^\prime$)",
#     "p_bw_MLE": r"bw_plan($x$)+mf",
#     "c_bw_MLE": r"bw_plan($x^\prime$)",
#     "p_fw_MLE": r"fw_plan($x$)",
#     "c_fw_MLE": r"fw_plan($x^\prime$)+mf",
#     "p_true_fw": r"fw_plan($P^*;x$)+mf",
#     "c_true_fw": r"fw_plan($P^*;x^\prime$)+mf",
#     "p_true_bw": r"bw_plan($\overleftarrow{P}^*;x$)+mf",
#     "c_true_bw": r"bw_plan($\overleftarrow{P}^*;x^\prime$)+mf",
#     "p_random_bw": r"bw_plan(unif $\overleftarrow{P},r^*;x$)+mf",
#     "c_random_bw": r"bw_plan(unif $\overleftarrow{P},r^*;x^\prime$)+mf",
#     "mb_p_random_bw": r"bw_plan(unif $\overleftarrow{P}r^*;x$)",
#     "mb_c_random_bw": r"bw_plan(unif $\overleftarrow{P},r^*;x^\prime$)",
# }

naming = {
    "vanilla": r"model_free",
    "mb_c_bw_MLE": r"bw_plan($\overleftarrow{P}$)",
    "mb_p_fw_MLE": r"fw_plan($P$)",
    "mb_c_true_bw": r"bw plan($\overleftarrow{P}^*$)",
    "mb_p_true_fw": r"fw_plan($P^*$)",
    "c_bw_MLE": r"bw_plan($\overleftarrow{P}$)",
    "p_fw_MLE": r"fw_plan($P^*$)",
}

# dotted = ["true_bw", "true_fw", "mb_true_fw", "mb_true_bw",
#           "true_bw_recur", "mb_true_bw_recur"]

all_agents = {
        "mb": ["mb_c_bw_MLE",
              "mb_p_fw_MLE",
              "mb_c_true_bw",
              "mb_p_true_fw",
              ],
        "mfmb": ["c_bw_MLE",
              "p_fw_MLE",
              ],
    }

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    plots_dir = os.path.join(FLAGS.plots, FLAGS.plotting)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig, ax = plt.subplots(1,
                           2,
                           squeeze=True,
                           figsize=(12, 5),
                           )
    all_data = {}

    for j, p in enumerate(plottings[FLAGS.plotting]):
        plot_config = plot_configs[p]
        all_agents_for_plotting = all_agents[p]
        colors = ["C{}".format(c) for c in range(len(all_agents_for_plotting))]
        unique_color_configs = [c for c in all_agents_for_plotting
                                if c not in dashed.keys()]

        alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}
        ax[j].set_title(plot_config["title"], fontsize=FONTSIZE)
        config_data = {}
        for i, res in enumerate(plot_config["results"]):
            env = res["env"]
            logs_dir = os.path.join(best_hyperparam_folder, env)
            agents_data = get_aoc(env, res["pivoting"], res["mb"], res["mle"], logs_dir, alg_to_color)
            config_data[env] = agents_data
        all_data[p] = config_data


        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

    np.save(os.path.join(plots_dir, "auc_data.npy"), all_data)

def get_aoc(env, pivoting, mb, mle, logs_dir, alg_to_color):
    env_config, volatile_agent_config = load_env_and_volatile_configs(env)

    name = pivoting + "_" + "all"
    if mb:
        name = "mb_" + name
    if mle:
        name = name + "_mle"

    agents_data = {}

    comparison_config = configs.comparison_configs.configs[env][name]
    persistent_agent_config = configs.agent_config.config["vanilla"]

    xs, ys = get_aoc_for_agent("vanilla", env_config, persistent_agent_config,
                   0, 0, logs_dir)

    agents_data["model_free"] = np.array([xs, ys])

    for i, agent in enumerate(comparison_config["agents"]):
        persistent_agent_config = configs.agent_config.config[agent]
        xs, ys = get_aoc_for_agent(agent, env_config, persistent_agent_config,
                        1, 0, logs_dir)
        agents_data[agent] = np.array([xs, ys])

    return agents_data

def get_aoc_for_agent(agent, env_config, persistent_agent_config,
                   planning_depth, replay_capacity, logs):
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

    tf_size_guidance = {
        'compressedHistograms': 100000,
        'images': 0,
        'scalars': 200000,
        'histograms': 1,
        'tensors': 200000,
    }

    num_runs = space["env_config"]["num_runs"]
    control_num_episodes = space["env_config"]["num_episodes"]

    xs = []
    ys = []
    for seed in range(num_runs):
        #print("seed_{}_agent_{}".format(seed, space["crt_config"]["agent"]))
        logs = os.path.join(os.path.join(space["crt_config"]["logs"],
                                         "summaries"),
                                        "seed_{}".format(seed))
        list_of_files = glob.glob(os.path.join(logs, '*'))  # * means all if need specific format then *.csv
        if len(list_of_files) == 0:
            print("no files in folder {}".format(logs))
            continue
        if len(list_of_files) > 1:
            print("ERROR, there should be only one file in folder {}".format(logs))
        filename = list_of_files[0]
        filepath = os.path.join(logs, filename)
        event_acc = EventAccumulator(filepath, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        # print(event_acc.Tags())
        tag = 'train/rmsve'
        if not tag in event_acc.Tags()["tensors"]:
            print("no tags")
            continue

        msve = event_acc.Tensors(tag)

        y = [tf.make_ndarray(m[2]) for m in msve]
        if len(y) == control_num_episodes:
            x = [m[1] for m in msve]
            xs.append(np.array(x))
            ys.append(np.array(y))


    return np.array(xs), np.array(ys)

if __name__ == '__main__':
    app.run(main)
