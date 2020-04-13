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
import glob
style.available
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
# style.use("classic")
plt.rcParams.update({'axes.titlesize': 'large'})
plt.rcParams.update({'axes.labelsize': 'large'})

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_string('plot_filename', "lp_explicit_exp", 'where to save results')
# flags.DEFINE_string('comarison_scheme', "fw_bw_gen", 'fw_bw or bw')
flags.DEFINE_string('comarison_scheme', "final", 'fw_bw or bw')
# flags.DEFINE_bool('nstep', True, 'n-step plot or comparison plt')
flags.DEFINE_bool('nstep', False, 'n-step plot or comparison plt')
flags.DEFINE_bool('cumulative_rmsve', False, 'n-step plot or comparison plt')
# flags.DEFINE_bool('cumulative_rmsve', True, 'n-step plot or comparison plt')
# flags.DEFINE_bool('all', True, 'n-step plot or comparison plt')
flags.DEFINE_bool('all', False, 'n-step plot or comparison plt')
flags.DEFINE_integer('num_runs', 10, '')
# flags.DEFINE_integer('num_runs', 1, '')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
flags.DEFINE_string('model_class', 'linear', 'tabular or linear')
# flags.DEFINE_string('model_class', 'tabular', 'tabular or linear')
flags.DEFINE_string('mdp_type', 'episodic', 'episodic or absorbing')
flags.DEFINE_string('env_type', 'discrete', 'discrete or continuous')
# flags.DEFINE_string('env_type', 'continuous', 'discrete or continuous')
# flags.DEFINE_string('obs_type', 'tabular', 'onehot, tabular, tile for continuous')
flags.DEFINE_string('obs_type', 'onehot', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('obs_type', 'tile', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('mdp', './continuous_mdps/obstacle.mdp',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
flags.DEFINE_string('mdp', './mdps/maze_48.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_486.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_864.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_80.mdp',
# flags.DEFINE_string('mdp', 'repeat',
# flags.DEFINE_string('mdp', 'loop',
# flags.DEFINE_string('mdp', 'loopy_chain',
# flags.DEFINE_string('mdp', 'tree',
# flags.DEFINE_string('mdp', 'shortcut',
# flags.DEFINE_string('mdp', 'random_chain',
# flags.DEFINE_string('mdp', 'po',
# flags.DEFINE_string('mdp', 'bandit',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_boolean('stochastic', False, 'stochastic transition dynamics or not.')
# flags.DEFINE_boolean('stochastic', True, 'stochastic transition dynamics or not.')
FLAGS = flags.FLAGS
FONTSIZE = 25
LINEWIDTH = 4
NON_GRIDWORLD_MDPS = ["random_chain", "boyan_chain", "bandit", "shortcut",
                      "loop", "tree", "repeat", "serial",
                      "po"]

naming_convention = {"bw":
                    {"tabular":
                        {"pred_exp":  'distribution/implicit',
                         "pred_gen":  'generative/implicit',
                         "jumpy_exp":  'distribution/explicit',
                         "jumpy_gen":  'generative/explicit',
                         "jumpy_fw_bw_gen":  'generative/search control',
                         "vanilla": "vanilla"},
                       "linear":
                        {"pred_exp":  'expectation/implici',
                         "jumpy_exp":  'expectation/explicit',
                         "vanilla": 'vanilla'
                       }
                      },
                    "fw_bw":
                            {"tabular":
                                {"explicit_exp":  'distribution/bw',
                                 "explicit_gen":  'generative/bw',
                                 "explicit_true":  'true_sample/bw',
                                 "explicit_iterat":  'iterate_generative/bw',
                                 "fw":  'distribution/fw_online',
                                 "fw_rnd":  'distribution/fw_random_buff',
                                 "fw_pri":  'distribution/fw_prioritized_buff',
                                 "bw_fw_exp":  'distribution/bw_fw',
                                 "bw_fw_gen":  'generative/bw_fw',
                                 "fw_bw_PWMA":  'distribution/fw_bw_buff_PWMA',
                                 "fw_bw_MG":  'distribution/fw_bw_buff_MG',
                                 "fw_bw_Imprv":  'distribution/fw_bw_buff_Imprv',
                                 },
                               "linear":
                                   {"jumpy_exp": 'expectation/backward',
                                    "jumpy_gen": 'generative/backward',
                                    "fw_exp": 'expectation/forward',
                                    "fw_gen": 'generative/forward'},
                              },
                     "final": {"tabular":
                          {"explicit_exp": 'distribution/bw',
                           "explicit_gen": 'generative/bw',
                           "explicit_true": 'true_sample/bw',
                           "explicit_iterat": 'iterate_generative/bw',
                           "fw": 'distribution/fw_online',
                           "fw_rnd": 'distribution/fw_random_buff',
                           "fw_pri": 'distribution/fw_prioritized_buff',
                           "bw_fw_exp": 'distribution/bw_fw',
                           "bw_fw_gen": 'generative/bw_fw',
                           "fw_bw_PWMA": 'distribution/fw_bw_buff_PWMA',
                           "fw_bw_MG": 'distribution/fw_bw_buff_MG',
                           "fw_bw_Imprv": 'distribution/fw_bw_buff_Imprv',
                           },
                      "linear":
                          {"explicit_exp": 'expectation/bw',
                           "explicit_gen": 'generative/bw',
                           "explicit_true": 'true_sample/bw',
                           "explicit_iterat": 'iterate_generative/bw',
                           "fw": 'distribution/fw_online',
                           "fw_rnd": 'expectation/fw_random_buff',
                           "fw_pri": 'expectation/fw_prioritized_buff',
                           "bw_fw_exp": 'expectation/bw_fw',
                           "bw_fw_gen": 'generative/bw_fw',
                           "fw_bw_PWMA": 'expectation/fw_bw_buff_PWMA',
                           "fw_bw_MG": 'expectation/fw_bw_buff_MG',
                           "fw_bw_Imprv": 'expectation/fw_bw_buff_Imprv',
                           },
                      }
                        }


def print_name(run_mode):
    if run_mode == "vanilla":
        return run_mode
    prefix_suffix = run_mode.split("_")
    suffix = prefix_suffix[-1]
    prefix = "_".join(prefix_suffix[:-1])
    return "{}_{}".format(naming_convention[FLAGS.comarison_scheme][FLAGS.model_class][prefix], suffix)

def main(argv):
    del argv  # Unused.
    mdp_filename = os.path.splitext(os.path.basename(FLAGS.mdp))[0]
    logs = os.path.join(FLAGS.logs, FLAGS.model_class)
    plots = os.path.join(FLAGS.plots, FLAGS.model_class)
    logs = os.path.join(logs, mdp_filename)
    plots = os.path.join(plots, mdp_filename)
    logs = os.path.join(logs, FLAGS.mdp_type)
    plots = os.path.join(plots, FLAGS.mdp_type)
    logs = os.path.join(logs, "stochastic" if FLAGS.stochastic else "deterministic")
    plots = os.path.join(plots, "stochastic" if FLAGS.stochastic else "deterministic")

    folders = [[int(run_mode_folder.split("_")[-1][1:]), run_mode_folder, os.path.join(logs, run_mode_folder), "-"]
                for run_mode_folder in os.listdir(logs)
                if run_mode_folder != ".DS_Store" and
                run_mode_folder != "vanilla" and
                not os.path.isfile(os.path.join(logs, run_mode_folder))]
    n = len(folders)

    if not FLAGS.all:
        folders.sort(reverse=True)
        if FLAGS.nstep:
            # prefix_suffix = FLAGS.plot_filename.split("_")
            # prefix = "_".join(prefix_suffix[-2:])
            # if prefix == "pred_exp":
            color = plt.cm.Blues(np.linspace(0.5, 1.0, n)[::-1])
            # elif prefix == "pred_gen":
            #     color = plt.cm.winter(np.linspace(0.3, 1.0, n))#[::-1])
            # elif prefix == "jumpy_exp":
            #     color = plt.cm.winter(np.linspace(0.3, 1.0, n))#[::-1])
            # elif prefix == "jumpy_gen":
            #     color = plt.cm.winter(np.linspace(0.3, 1.0, n))#[::-1])
            hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                           tuple(color[:, 0:-1]))
            color = hexcolor  # plt.cm.viridis(np.linspace(0, 1, n))
            mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
            colors = None
        else:

            all_colors = {"bw":
                          {"pred_exp":  'midnightblue',
                           "pred_gen":  'deepskyblue',
                           "jumpy_exp":  'lightseagreen',
                           "jumpy_gen":  'turquoise'},
                       "fw_bw":
                           {
                               # "fw":  'midnightblue',
                            # "fw_gen": 'deepskyblue',

                            "bw_fw_exp": 'darkorange',
                            "bw_fw_gen": 'gold',

                            # "fw_rnd": 'purple',
                            # "fw_pri": 'violet',
                            # "fw_bw_PWMA": 'mediumvioletred',
                            # "fw_bw_MG": 'deeppink',
                            # "fw_bw_Imprv": 'pink',

                            "fw_rnd": 'purple',
                            "fw_pri": 'violet',
                            "fw_bw_PWMA": 'mediumvioletred',
                            "fw_bw_MG": 'deeppink',
                            "fw_bw_Imprv": 'pink',

                            "explicit_exp": 'lightseagreen',
                            "explicit_gen": 'turquoise',
                            "explicit_true": 'black',

                            "explicit_iterat": 'chocolate',
                            },
                        "final":{
                            "explicit_exp": 'teal',
                            "explicit_gen": 'blue',
                            "bw_fw_exp": 'deepskyblue',
                            "fw_rnd": 'green',
                            "fw_pri": 'olive',
                            "fw_bw_PWMA": 'blue',
                            "fw_bw_MG": 'darkblue',
                            "fw_bw_Imprv": 'black',
                            "explicit_iterat": 'cyan',
                            "explicit_true": 'slategray',

                        }
                      }
            colors = all_colors[FLAGS.comarison_scheme]

        for i in range(len(folders)):
            _, run_mode_folder, folder_path, linestyle = folders[i]
            plot_tensorflow_log(folder_path, run_mode_folder, linestyle, colors)

        plot_tensorflow_log(os.path.join(logs, "vanilla"), "vanilla", ':', None)

        plt.xlabel("Episode count", fontsize=FONTSIZE)

        if FLAGS.mdp in NON_GRIDWORLD_MDPS:
            if FLAGS.cumulative_rmsve:
                yaxis = 'Cumulative RMSVE'
            else:
                yaxis = 'RMSVE'
        else:
            yaxis = 'MSVE'

        plt.ylabel(yaxis, fontsize=FONTSIZE)
        plt.legend(loc='lower right' if FLAGS.cumulative_rmsve else 'upper right',
                   frameon=True,
                   prop={'size': FONTSIZE})
        # plt.show()
        if not os.path.exists(plots):
            os.makedirs(plots)

        plt.savefig(os.path.join(plots,
                                 "{}_{}.png".format(FLAGS.plot_filename,
                                                    "CumRMSVE" if
                                                    FLAGS.cumulative_rmsve else
                                                    "RMSVE")))
    else:

        folders_pred_exp = [f for f in folders if f[1].startswith("pred_exp")]
        folders_pred_gen = [f for f in folders if f[1].startswith("pred_gen")]
        folders_jumpy_exp = [f for f in folders if f[1].startswith("jumpy_exp")]
        folders_jumpy_gen = [f for f in folders if f[1].startswith("jumpy_gen")]

        n = np.max([len(folders_pred_exp), len(folders_pred_gen),
                          len(folders_jumpy_exp), len(folders_jumpy_gen)])
        colors = {"pred_exp": plt.cm.Blues(np.linspace(0.5, 0.9, n)[::-1]),
                  "pred_gen": plt.cm.YlGn(np.linspace(0.5, 0.9, n)[::-1]),
                  "jumpy_exp": plt.cm.RdPu(np.linspace(0.5, 0.9, n)[::-1]),
                  "jumpy_gen": plt.cm.Wistia(np.linspace(0.5, 0.9, n)[::-1]),
                  }
        linestyles = {
            "pred_exp":  '-',
            "pred_gen":  '--',
            "jumpy_exp":  ':',
            "jumpy_gen":  '-.'
        }
        all_folders = {"pred_exp": folders_pred_exp,
                       "pred_gen": folders_pred_gen,
                       "jumpy_exp": folders_jumpy_exp,
                       "jumpy_gen": folders_jumpy_gen}
        for k, folders in all_folders.items():
            folders.sort(reverse=True)

            color = colors[k]  # This returns RGBA; convert:
            hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                           tuple(color[:, 0:-1]))
            color = hexcolor  # plt.cm.viridis(np.linspace(0, 1, n))
            # mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
            ax = plt.gca()
            ax.set_prop_cycle(cycler.cycler('color', color))

            for i in range(len(folders)):
                _, run_mode_folder, folder_path = folders[i]
                plot_tensorflow_log(folder_path, run_mode_folder, linestyles[k])

        _, run_mode_folder, folder_path = [0, "vanilla", os.path.join(logs, "vanilla")]
        plot_tensorflow_log(folder_path, run_mode_folder, "-")

        plt.xlabel("Episode count", fontsize=FONTSIZE)

        if FLAGS.mdp in NON_GRIDWORLD_MDPS:
            if FLAGS.cumulative_rmsve:
                yaxis = 'Cumulative RMSVE'
            else:
                yaxis = 'RMSVE'
        else:
            yaxis = 'MSVE'

        plt.ylabel(yaxis, fontsize=FONTSIZE)
        plt.legend(loc='upper right', frameon=True, prop={'size': FONTSIZE})
        # plt.show()
        if not os.path.exists(plots):
            os.makedirs(plots)

        plt.savefig(os.path.join(plots,
                                 "{}_{}.png".format(FLAGS.plot_filename,
                                                    "CumRMSVE" if
                                                    FLAGS.cumulative_rmsve else
                                                    "RMSVE")))


def plot_tensorflow_log(path, run_mode, linestyle, colors=None):
    tf_size_guidance = {
        'compressedHistograms': 100000,
        'images': 0,
        'scalars': 200000,
        'histograms': 1,
        'tensors': 200000,
    }
    mean_y_over_seeds = None
    all_y_over_seeds = []
    for seed in range(FLAGS.num_runs):
        logs = os.path.join(os.path.join(path, "summaries"), "seed_{}".format(seed))
        list_of_files = glob.glob(os.path.join(logs, '*'))  # * means all if need specific format then *.csv
        if len(list_of_files) == 0:
            print("no files")
            return
        # filename = max(list_of_files, key=os.path.getctime)
        if len(list_of_files) > 1:
            print("ERROR, there should be only one file")
        filename = list_of_files[0]
        # filename = os.listdir(logs)[0]
        filepath = os.path.join(logs, filename)
        event_acc = EventAccumulator(filepath, tf_size_guidance)
        event_acc.Reload()


        # Show all tags in the log file
        print(event_acc.Tags())
        if FLAGS.mdp in NON_GRIDWORLD_MDPS:
            if FLAGS.cumulative_rmsve:
                tag = 'train/cumulative_rmsve'
            else:
                tag = 'train/rmsve'
        else:
            tag = 'train/msve'
        if not tag in event_acc.Tags()["tensors"]:
            return

        msve = event_acc.Tensors(tag)

        # steps = len(msve)
        # x = np.arange(steps)
        x = [m[1] for m in msve]
        y = [tf.make_ndarray(m[2]) for m in msve]
        # if mean_y_over_seeds is None:
        #     mean_y_over_seeds = np.zeros_like(y)
        # mean_y_over_seeds += 1/FLAGS.num_runs + np.array(y)
        all_y_over_seeds.append(np.array(y))

    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    if run_mode == "vanilla":
        plt.plot(x, mean_y_over_seeds, label=print_name(run_mode), c="r", alpha=1, linewidth=LINEWIDTH, linestyle=linestyle)#, marker='v')
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         color="r", alpha=0.1)
    else:
        if FLAGS.nstep == False:
            prefix_suffix = run_mode.split("_")
            prefix = "_".join(prefix_suffix[:-1])
            plt.plot(x, mean_y_over_seeds, label=print_name(run_mode), alpha=1, linewidth=LINEWIDTH,
                     linestyle=linestyle, c=colors[prefix])#, marker='v')
            plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                             alpha=0.1, color=colors[prefix])
        else:
            plt.plot(x, mean_y_over_seeds, label=print_name(run_mode), alpha=1, linewidth=LINEWIDTH,
                     linestyle=linestyle)  # , marker='v')
            plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                             alpha=0.1)

if __name__ == '__main__':
    app.run(main)
