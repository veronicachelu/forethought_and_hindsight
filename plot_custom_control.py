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

flags.DEFINE_string('logs', os.path.join(str((os.environ['LOGS'])), 'control'), 'where to save results')
flags.DEFINE_string('config', "maze", 'where to save results')
flags.DEFINE_bool('tabular', False, 'where to save results')
flags.DEFINE_bool('mb', False, 'where to save results')
flags.DEFINE_bool('reward', False, 'where to save results')
# flags.DEFINE_bool('reward', False, 'where to save results')
# flags.DEFINE_bool('mb', False, 'where to save results')
flags.DEFINE_string('pivoting', "control", 'where to save results')
flags.DEFINE_float('ymin', None, 'plot up to')
flags.DEFINE_float('ymax', None, 'plot up to')
flags.DEFINE_integer('max', None, 'plot up to')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')

FLAGS = flags.FLAGS
FONTSIZE = 25
LINEWIDTH = 3
TICKSIZE = 15

plot_configs = {
    "maze": {
        "nr": 2,
        "nc": 2,
        "subplots":
        [
            {
                "env": "maze_1",
                "pivoting": "pc",
                "title": "Deterministic",
            },
            {
                "env": "maze_stoch",
                "pivoting": "pc",
                "title": "Stochastic transitions \n (random move w/ p=0.5)"
            },
            {
                "env": "maze_05",
                "pivoting": "pc",
                "title": "Stochastic reward \n (r(G)=+1 w/ p=0.5)"
            },
            {
                "env": "maze_01",
                "pivoting": "pc",
                "title": "Stochastic reward \n (r(G)=+1 w/ p=0.1)",
            },
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
          "c_bw_q": "p_bw_q",
          "c_fw_q": "p_fw_q",
          "c_true_fw_q": "p_true_fw_q",
          "p_bw_PAML": "p_bw",
          "c_bw_PAML": "c_bw",
          "p_fw_PAML": "p_fw",
          "c_fw_PAML": "c_fw",

          "c_ac_bw_PAML": "c_ac_bw",
          "p_ac_fw_PAML": "p_ac_fw",

          # "p_bw_PAML_MLE": "p_bw",
          # "c_bw_PAML_MLE": "c_bw",
          # "p_fw_PAML_MLE": "p_fw",
          # "c_fw_PAML_MLE": "c_fw",
          # "p_true_bw_recur": "p_bw_recur",
          # "c_true_bw_recur": "c_bw_recur",

          }

dotted = {
        "p_true_fw_q": "p_fw_q",
}

all_agents = [
              # "p_bw_q",
              "p_bw_q",
              "p_fw_q",
              "p_true_fw_q",
              "p_true_fw_q",
              # "c_fw_q",
              "p_bw_q_1",
              # "p_bw_q_2",
              # "p_bw_q_3"
]

# naming = {
#     "q": r"model_free(mf)",
#     "p_bw_q": r"bw_plan($\mathbf{x},a,{x}^\prime$)+mf",
#     "p_bw_q_1": r"bw_plan($\mathbf{x},a,{x}^\prime;top\_1$)+mf",
#     "p_bw_q_2": r"bw_plan($\mathbf{x},a,{x}^\prime;top\_2$)+mf",
#     "p_bw_q_3": r"bw_plan($\mathbf{x},a,{x}^\prime;top\_3$)+mf",
#     "c_bw_q": r"bw_plan(${x},a,\mathbf{x}^\prime$)+mf",
#     "p_fw_q": r"fw_plan($\mathbf{x},a,{x}^\prime$)+mf",
#     "c_fw_q": r"fw_plan(${x},a,\mathbf{x}^\prime$)+mf",
#     "p_true_fw_q": r"fw_plan($P^*;\mathbf{x},a,{x}^\prime$)+mf",
#     "c_true_fw_q": r"fw_plan($P^*;{x},a,\mathbf{x}^\prime$)+mf",
# }
naming = {
    "q": r"model_free",
    "p_bw_q": r"bw_plan($\overleftarrow{P}$)",
    "p_bw_q_1": r"bw_plan($\overleftarrow{P}\downarrow[0]$)",
    "p_fw_q": r"fw_plan($P$)",
    "p_true_fw_q": r"fw_plan($P^*$)",
}

def main(argv):
    del argv  # Unused.

    best_hyperparam_folder = FLAGS.logs

    plots_dir = os.path.join(FLAGS.plots, FLAGS.config)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    yaxis = 'Steps/Episode'
    xaxis = "Episodes"
    fig, ax = plt.subplots(plot_configs[FLAGS.config]["nr"],
                           plot_configs[FLAGS.config]["nc"],
                           sharex='col',
                           sharey='row',
                           squeeze=True,
                           figsize=(12, 8),
                           )
    # ax.set(aspect="auto")
    unique_color_configs = [c for c in all_agents
                            if c not in dashed.keys()]

    colors = ["C{}".format(c) for c in range(len(all_agents))]
    alg_to_color = {alg: color for alg, color in zip(unique_color_configs, colors)}

    # fig.set_title("Learning, planning & acting", fontsize=FONTSIZE)
    all_handles = []
    all_labels = []
    for ii, sub in enumerate(plot_configs[FLAGS.config]["subplots"]):
        # ii, jj = np.unravel_index(i, (plot_configs[FLAGS.config]["nr"],
        #                    plot_configs[FLAGS.config]["nc"]))
        i, j = np.unravel_index(ii, (plot_configs[FLAGS.config]["nr"],
                                     plot_configs[FLAGS.config]["nc"]))
        max = sub["max"] if "max" in sub.keys() else None
        env = sub["env"]
        if "title" in sub.keys():
            ax[i, j].set_title(sub["title"], fontsize=FONTSIZE)
        logs_dir = os.path.join(best_hyperparam_folder, env)

        if j == 0:
            ax[i, j].set_ylabel(yaxis, fontsize=FONTSIZE)

        plt.setp(ax[i, j].get_yticklabels(), visible=True, fontsize=TICKSIZE)
        # if ii == plot_configs[FLAGS.config]["nr"] - 1:
        if i == plot_configs[FLAGS.config]["nr"] - 1:
            ax[i, j].set_xlabel(xaxis, fontsize=FONTSIZE)
        plt.setp(ax[i, j].get_xticklabels(), visible=True, fontsize=TICKSIZE)

        ax[i, j].grid(True)

        lines = plot(env, sub["pivoting"], logs_dir, plots_dir, ax[i, j], max, alg_to_color)
        handles, labels = ax[i, j].get_legend_handles_labels()
        all_handles.extend(handles)
        all_labels.extend(labels)

    # ax1.set_xticklabels([])
    # Set the y limits
    # if FLAGS.ymin is not None and FLAGS.ymax is not None:
    #     plt.ylim(FLAGS.ymin, FLAGS.ymax)
    # plt.figlegend(
    # *[*zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1]
    # by_label = dict(zip(labels, handles))
    # all_labels = by_label.values()
    # all_handles = by_label.keys()
    fig.legend(
        # handles=all_handles,
        # labels=all_labels,
        *[*zip(*{l: h for h, l in zip(all_handles, all_labels)}.items())][::-1],
        # loc='lower right' if FLAGS.cumulative_rmsve else 'upper right',
        frameon=False,
    #     ncol=plot_configs[FLAGS.config]["nc"],
    #     mode="expand",
    #     # bbox_to_anchor=(0.0, -0.05, 1.0, 0.3),
    #     # loc='upper left',
    #     # loc="lower right",
    #     loc="upper left",
    #     bbox_to_anchor=(0, 0, 1.0, 0.2),
    #     # bbox_transform=fig.transFigure,
    #     # borderaxespad=0.,
    #     prop={'size': FONTSIZE},
    #     # bbox_to_anchor=(0., 1.0, 1.0, 0.1)
      prop={'size': FONTSIZE},
        bbox_to_anchor=(1.03, 0.5),
        loc="upper center",
        # bbox_to_anchor=(0.5, -0.05)#, 1.0, 0.1)
        # bbox_to_anchor=(1., 1.)#, 1.0, 0.1)
        # bbox_to_anchor=(0., 1.0, 1.0, 0.1)

    )
    # )
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=5)
    # bbox_to_anchor=(1.1, 1.1))
    # plt.legend(loc='upper right',
    #            frameon=True,
    #            prop={'size': FONTSIZE},
    #            bbox_to_anchor=(1.1, 1.1))
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # name = pivoting
    # if FLAGS.mb:
    #     name = "mb_" + name
    # if FLAGS.reward:
    #     name = name + "_reward"
    # plt.show()
    # fig.set_grid()
    fig.tight_layout()
    fig.subplots_adjust(right=0.90)
    fig.savefig(os.path.join(plots_dir,
                             "{}_{}.png".format("all",
                                                "steps")),
        bbox_inches = 'tight',
        )


def plot(env, pivoting, logs_dir, plots_dir, ax, max, alg_to_color):

    env_config, volatile_agent_config = load_env_and_volatile_configs(env)

    comparison_config = configs.comparison_configs.configs[env][pivoting]
    plot_for_agent("q", env_config, logs_dir, "gray", "-", max, ax)

    lines = []
    for i, agent in enumerate(comparison_config["agents"]):
        if agent not in dotted.keys():
            color = alg_to_color[agent]
            linestyle = "-"
        else:
            color = alg_to_color[dotted[agent]]
            linestyle = ":"

        line = plot_for_agent(agent, env_config, logs_dir, color, linestyle, max, ax)
        lines.append(line)

    return lines

def plot_for_agent(agent, env_config,
                logs_dir, color, linestyle, max, ax):
    print(agent)
    log_folder_agent = os.path.join(logs_dir, "{}".format(agent))
    volatile_config = {"agent": agent,
                       "logs_dir": log_folder_agent}
    space = {
    "env_config": env_config,
    "crt_config": volatile_config}
    line = plot_tensorflow_log(space, color, linestyle, max, ax)
    return line

def plot_tensorflow_log(space, color, linestyle, max, ax):
    tf_size_guidance = {
        'compressedHistograms': 100000,
        'images': 0,
        'scalars': 200000,
        'histograms': 1,
        'tensors': 200000,
    }
    all_y_over_seeds = []
    all_x_over_seeds = []
    the_incomplete = []
    num_runs = space["env_config"]["num_runs"]
    control_num_episodes = space["env_config"]["control_num_episodes"]
    for seed in range(num_runs):
        #print("seed_{}_agent_{}".format(seed, space["crt_config"]["agent"]))
        logs_dir = os.path.join(os.path.join(space["crt_config"]["logs_dir"],
                                         "summaries"),
                                        "seed_{}".format(seed))
        list_of_files = glob.glob(os.path.join(logs_dir, '*'))  # * means all if need specific format then *.csv
        if len(list_of_files) == 0:
            print("no files in folder {}".format(logs_dir))
            continue
        if len(list_of_files) > 1:
            print("ERROR, there should be only one file in folder {}".format(logs_dir))
        filename = list_of_files[0]
        filepath = os.path.join(logs_dir, filename)
        event_acc = EventAccumulator(filepath, tf_size_guidance)
        event_acc.Reload()

        # Show all tags in the log file
        # print(event_acc.Tags())

        tag = 'train/steps'
        if not tag in event_acc.Tags()["tensors"]:
            print("no tags")
            continue

        msve = event_acc.Tensors(tag)
        y_steps = [tf.make_ndarray(m[2]) for m in msve]

        if len(y_steps) == control_num_episodes:
            x = [m[1] for m in msve]
            all_y_over_seeds.append(np.array(y_steps))

    # max_size = np.max([len(a) for a in all_y_over_seeds])
    # the_incomplete_seeds = [i for i, a in enumerate(all_y_over_seeds) if len(a) != max_size]
    # print(the_incomplete_seeds)
    # all_y_over_complete_seeds = [a for i, a in enumerate(all_y_over_seeds) if len(a) == max_size]
    # the_complete_seeds = [i for i, a in enumerate(all_y_over_seeds) if len(a) == max_size]
    #
    if len(all_y_over_seeds) == 0:
        print("agent_{} has no data!".format(space["crt_config"]["agent"]))
        return

    # x = all_x_over_seeds[the_complete_seeds[0]]
    # x = all_x_over_seeds[the_complete_seeds[0]]
    # the_complete = [a for i, a in enumerate(all_y_over_seeds) if len(a) == first_seed_size]
    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    ste_y_over_seeds = np.divide(std_y_over_seeds, np.sqrt(num_runs))
    # mean_y_over_seeds = mean_y_over_seeds[::5]
    # std_y_over_seeds = std_y_over_seeds[::5]
    # x = x[::5]
    if max is not None:
        x = x[:max]
        mean_y_over_seeds = mean_y_over_seeds[:max]
        std_y_over_seeds = std_y_over_seeds[:max]
        ste_y_over_seeds = np.divide(std_y_over_seeds, np.sqrt(num_runs))

    # ste_y_over_seeds *= 2
    label = naming[space["crt_config"]["agent"]]
    if space["crt_config"]["agent"] == "q":
        line = ax.plot(x, mean_y_over_seeds, label=label, c="gray", alpha=1, linewidth=LINEWIDTH, linestyle="-")
        ax.fill_between(x, mean_y_over_seeds - ste_y_over_seeds, mean_y_over_seeds + ste_y_over_seeds,
                         color="gray", alpha=0.2)
    else:
        line = ax.plot(x, mean_y_over_seeds, label=label,
                 alpha=1, linewidth=LINEWIDTH, color=color,
                 linestyle=linestyle)
        ax.fill_between(x, mean_y_over_seeds - ste_y_over_seeds, mean_y_over_seeds + ste_y_over_seeds,
                         alpha=0.2, color=color,
                         linestyle=linestyle)

    return line

if __name__ == '__main__':
    app.run(main)
