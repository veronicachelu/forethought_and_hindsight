import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import tensorflow as tf
import matplotlib as mpl
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
flags.DEFINE_string('plot_filename', "lp_nstep_jumpy_exp", 'where to save results')
flags.DEFINE_bool('nstep', True, 'n-step plot or comparison plt')
# flags.DEFINE_bool('nstep', False, 'n-step plot or comparison plt')
# flags.DEFINE_bool('all', True, 'n-step plot or comparison plt')
flags.DEFINE_bool('all', False, 'n-step plot or comparison plt')
# flags.DEFINE_integer('num_runs', 20, '')
flags.DEFINE_integer('num_runs', 5, '')
# flags.DEFINE_bool('nstep', True, 'n-step plot or comparison plt')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
# flags.DEFINE_string('run_mode', 'nstep', 'what agent to run')
flags.DEFINE_string('model_class', 'linear', 'tabular or linear')
# flags.DEFINE_string('model_class', 'tabular', 'tabular or linear')
flags.DEFINE_string('mdp_type', 'episodic', 'episodic or absorbing')
# flags.DEFINE_string('env_type', 'discrete', 'discrete or continuous')
flags.DEFINE_string('env_type', 'continuous', 'discrete or continuous')
# flags.DEFINE_string('obs_type', 'tabular', 'onehot, tabular, tile for continuous')
# flags.DEFINE_string('obs_type', 'onehot', 'onehot, tabular, tile for continuous')
flags.DEFINE_string('obs_type', 'tile', 'onehot, tabular, tile for continuous')
flags.DEFINE_string('mdp', './continuous_mdps/obstacle.mdp',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_486.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_864.mdp',
# flags.DEFINE_string('mdp', './mdps/maze_80.mdp',
# flags.DEFINE_string('mdp', 'random_chain',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_boolean('stochastic', False, 'stochastic transition dynamics or not.')
# flags.DEFINE_boolean('stochastic', True, 'stochastic transition dynamics or not.')
FLAGS = flags.FLAGS
FONTSIZE = 25
LINEWIDTH = 4

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
            color = plt.cm.Blues(np.linspace(0.5, 1.0, n)[::-1])  # This returns RGBA; convert:
        else:
            color = plt.cm.winter(np.linspace(0.0, 1.0, n)[::-1])  # This returns RGBA; convert:
        hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                       tuple(color[:, 0:-1]))
        color = hexcolor #plt.cm.viridis(np.linspace(0, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

        folders.append([0, "vanilla", os.path.join(logs, "vanilla"), ':'])
        for i in range(len(folders)):
            _, run_mode_folder, folder_path, linestyle = folders[i]
            plot_tensorflow_log(folder_path, run_mode_folder, linestyle)

        plt.xlabel("Episode count", fontsize=FONTSIZE)

        if FLAGS.mdp == "random_chain" or FLAGS.mdp == "boyan_chain":
            yaxis = 'RMSVE'
        else:
            yaxis = 'MSVE'

        plt.ylabel(yaxis, fontsize=FONTSIZE)
        plt.legend(loc='upper right', frameon=True, prop={'size': FONTSIZE})
        # plt.show()
        if not os.path.exists(plots):
            os.makedirs(plots)

        plt.savefig(os.path.join(plots, "{}.png".format(FLAGS.plot_filename)))
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

        if FLAGS.mdp == "random_chain" or FLAGS.mdp == "boyan_chain":
            yaxis = 'RMSVE'
        else:
            yaxis = 'MSVE'

        plt.ylabel(yaxis, fontsize=FONTSIZE)
        plt.legend(loc='upper right', frameon=True, prop={'size': FONTSIZE})
        # plt.show()
        if not os.path.exists(plots):
            os.makedirs(plots)

        plt.savefig(os.path.join(plots, "{}.png".format(FLAGS.plot_filename)))


def plot_tensorflow_log(path, run_mode, linestyle):
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
        if FLAGS.mdp == "random_chain" or FLAGS.mdp == "boyan_chain":
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
        if mean_y_over_seeds is None:
            mean_y_over_seeds = np.zeros_like(y)
        mean_y_over_seeds += 1/FLAGS.num_runs + np.array(y)
        all_y_over_seeds.append(np.array(y))

    mean_y_over_seeds = np.mean(all_y_over_seeds, axis=0)
    std_y_over_seeds = np.std(all_y_over_seeds, axis=0)
    if run_mode == "vanilla":
        plt.plot(x, mean_y_over_seeds, label=run_mode, c="r", alpha=1, linewidth=LINEWIDTH, linestyle=linestyle)#, marker='v')
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         color="r", alpha=0.2)
    else:
        plt.plot(x, mean_y_over_seeds, label=run_mode, alpha=1, linewidth=LINEWIDTH, linestyle=linestyle)#, marker='v')
        plt.fill_between(x, mean_y_over_seeds - std_y_over_seeds, mean_y_over_seeds + std_y_over_seeds,
                         alpha=0.2)

if __name__ == '__main__':
    app.run(main)
