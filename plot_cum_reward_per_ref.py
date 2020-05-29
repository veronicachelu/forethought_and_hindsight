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
from matplotlib.ticker import FuncFormatter
style.available
style.use('seaborn-poster') #sets the size of the charts
# style.use('ggplot')
style.use("default")
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
# flags.DEFINE_string('env', "bipartite_100_1", 'where to save results')
flags.DEFINE_string('config', "cum_reward", 'where to save results')
# flags.DEFINE_string('env', "fanin", 'where to save results')
flags.DEFINE_bool('tabular', True, 'where to save results')
flags.DEFINE_float('ymin', None, 'plot up to')
flags.DEFINE_float('ymax', None, 'plot up to')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
FLAGS = flags.FLAGS
FONTSIZE = 23
TICKSIZE = 15
LINEWIDTH = 3

plot_configs = {
    "cum_reward": {
        "nr": 1,
        "nc": 2,
        "subplots": [
            {
                "env": "maze_1",
                "pivoting": "ref",
                "title": "Learning \& Planning"
            },
            {
                "env": "maze_1",
                "pivoting": "mb_ref",
                "title": "Pure Planning"
            }
        ]
    }
}
dotted = {
          "c_bw_q": "p_bw_q",
          "c_fw_q": "p_fw_q",
          "mb_c_bw_q": "mb_p_bw_q",
          "mb_c_fw_q": "mb_p_fw_q",
}

naming = {
    "p_bw_q": r"bw_plan($\mathbf{x},a,{x}^\prime$)",
    "c_bw_q": r"bw_plan(${x},a,\mathbf{x}^\prime$)",
    "p_fw_q": r"fw_plan($\mathbf{x},a,{x}^\prime$)",
    "c_fw_q": r"fw_plan(${x},a,\mathbf{x}^\prime$)",
    "mb_p_bw_q": r"bw_plan($\mathbf{x},a,{x}^\prime$)",
    "mb_c_bw_q": r"bw_plan(${x},a,\mathbf{x}^\prime$)",
    "mb_p_fw_q": r"fw_plan($\mathbf{x},a,{x}^\prime$)",
    "mb_c_fw_q": r"fw_plan(${x},a,\mathbf{x}^\prime$)",
}
# dotted = ["true_bw", "true_fw", "mb_true_fw", "mb_true_bw",
#           "true_bw_recur", "mb_true_bw_recur"]
all_agents_per_config = {
    "ref": [
          "p_bw_q",
          "p_fw_q",
          ],
    "mb_ref": [
         "mb_p_bw_q",
         "mb_p_fw_q",
    ]
}

def main(argv):
    del argv  # Unused.
    best_hyperparam_folder = os.path.join(FLAGS.logs, "best")
    plots_dir = os.path.join(FLAGS.plots, FLAGS.config)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    yaxis = 'Cumulative Reward'
    xaxis = "Timesteps"

    fig, ax = plt.subplots(plot_configs[FLAGS.config]["nr"],
                           plot_configs[FLAGS.config]["nc"],
                           sharex='col',
                           squeeze=True,  # , sharey=True,
                           figsize=(12, 5),
                           )


    all_handles = []
    all_labels = []
    # f = lambda x, pos: f'{x/10**3:,.1f}K' if x >= 1000 else f'{x:,.0f}'
    for i, sub in enumerate(plot_configs[FLAGS.config]["subplots"]):
        unique_color_configs = [c for c in all_agents_per_config[sub["env"]]
                                if c not in dotted.keys()]
        colors = ["C{}".format(c) for c in range(len(all_agents_per_config[sub["env"]]))]
        alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}

        env = sub["env"]
        if "title" in sub.keys():
            ax[i].set_title(sub["title"], fontsize=FONTSIZE)
        logs_dir = os.path.join(best_hyperparam_folder, env)
        if i == 0:
            ax[i].set_ylabel(yaxis, fontsize=FONTSIZE)
        ax[i].set_xlabel(xaxis, fontsize=FONTSIZE)
        ax[i].grid(True)
        # Shrink current axis's height by 10% on the bottom
        # box = ax[i].get_position()
        # ax[i].set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])
        plt.setp(ax[i].get_yticklabels(), visible=True, fontsize=TICKSIZE)
        plt.setp(ax[i].get_xticklabels(), visible=True, fontsize=TICKSIZE)
        ax[i].ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        # ticklabel_format(axis='x', style='sci',
        #                        scilimits=None, useOffset=None, useLocale=None,
        #                  useMathText=True)
        # ax[i].xaxis.set_major_formatter(FuncFormatter(f))
        plot(env, sub["pivoting"], logs_dir, plots_dir, ax[i], alg_to_color)
        handles, labels = ax[i].get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    fig.legend(
        # handles=all_handles,
        # labels=all_labels,
        *[*zip(*{l: h for h, l in zip(all_handles, all_labels)}.items())][::-1],
        # loc='lower right' if FLAGS.cumulative_rmsve else 'upper right',
        frameon=False,
        # ncol=5,
        # mode="expand",
        # loc = 7,
        # loc='lower left',
        # loc='upper center',
        # loc='upper left',
        # borderaxespad=0.,
        prop={'size': FONTSIZE},
        bbox_to_anchor=(1.03, 0.8),
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.05)#, 1.0, 0.1)
        # bbox_to_anchor=(1., 1.)#, 1.0, 0.1)
        # bbox_to_anchor=(0., 1.0, 1.0, 0.1)

    )

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    fig.tight_layout()
    fig.subplots_adjust(right=0.90)
    fig.savefig(os.path.join(plots_dir,
                             "{}_{}.png".format("all",
                                                FLAGS.config)),
                bbox_inches='tight',
                )

def plot(env, pivoting, logs_dir, plots_dir, ax, alg_to_color):
    env_config, volatile_agent_config = load_env_and_volatile_configs(env)

    name = pivoting

    comparison_config = configs.comparison_configs.configs[env][name]

    # unique_color_configs = [c for c in comparison_config["agents"]
    #                         if c not in internal_dashed.keys()]
    # n = len(unique_color_configs)
    #
    # colors = ["C{}".format(c) for c in range(n)]
    # alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}

    persistent_agent_config = configs.agent_config.config["vanilla"]
    plot_for_agent("vanilla", env_config, persistent_agent_config,
                   0, 0, logs_dir, "gray", "-", ax)

    for i, agent in enumerate(comparison_config["agents"]):
        if agent not in dotted.keys():
            color = alg_to_color[agent]
            linestyle = "-"
        else:
            color = alg_to_color[dotted[agent]]
            linestyle = ":"

        persistent_agent_config = configs.agent_config.config[agent]
        plot_for_agent(agent, env_config, persistent_agent_config,
                        1, 0, logs_dir, color, linestyle, ax)


def plot_for_agent(agent, env_config, persistent_agent_config,
                   planning_depth, replay_capacity, logs, color, linestyle, ax):
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
    plot_tensorflow_log(space, color, linestyle, ax)

def plot_tensorflow_log(space, color, linestyle, ax):
    tf_size_guidance = {
        'compressedHistograms': 100000,
        'images': 0,
        'scalars': 200000,
        'histograms': 1,
        'tensors': 200000,
    }
    all_y_over_seeds = []
    num_runs = space["env_config"]["num_runs"]
    control_num_episodes = space["env_config"]["num_episodes"]

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
        if FLAGS.cumulative_rmsve:
            tag = 'train/total_rmsve'
        else:
            tag = 'train/rmsve'
        if not tag in event_acc.Tags()["tensors"]:
            print("no tags")
            continue

        msve = event_acc.Tensors(tag)

        y = [tf.make_ndarray(m[2]) for m in msve]
        if len(y) == control_num_episodes:
            x = [m[1] for m in msve]
            all_y_over_seeds.append(np.array(y))

    if len(all_y_over_seeds) == 0:
        print("agent_{} has no data!".format(space["crt_config"]["agent"]))
        return

    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    ste_y_over_seeds = np.divide(std_y_over_seeds, np.sqrt(num_runs))
    # std_y_over_seeds /= np.sqrt(len(std_y_over_seeds))
    label = naming[space["crt_config"]["agent"]]
    if space["crt_config"]["agent"] == "vanilla":
        g = ax.plot(x, mean_y_over_seeds, label=label, c="gray", alpha=1, linewidth=LINEWIDTH, linestyle="-")
        ax.fill_between(x, mean_y_over_seeds - ste_y_over_seeds, mean_y_over_seeds + ste_y_over_seeds,
                         color="gray", alpha=0.07)
    else:
        # if FLAGS.paml and space["crt_config"]["max_norm"] is not None:
        #     label += "_{}".format(space["crt_config"]["max_norm"])
        g = ax.plot(x, mean_y_over_seeds, label=label,
                 alpha=1, linewidth=LINEWIDTH, color=color,
                 linestyle=linestyle)
        ax.fill_between(x, mean_y_over_seeds - ste_y_over_seeds, mean_y_over_seeds + ste_y_over_seeds,
                         alpha=0.07, color=color,
                         linestyle=linestyle)

    # xlabels = ['{:,.2f}'.format(x) + 'K' for x in g.get_xticks() / 1000]
    # g.set_xticklabels(xlabels)

if __name__ == '__main__':
    app.run(main)