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
style.available
style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')

flags.DEFINE_string('logs', str((os.environ['LOGS'])), 'where to save results')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
# flags.DEFINE_string('run_mode', 'nstep', 'what agent to run')
flags.DEFINE_string('model_class', 'linear', 'tabular or linear')
# flags.DEFINE_string('model_class', 'tabular', 'tabular or linear')
flags.DEFINE_string('env_type', 'continuous', 'discrete or continuous')
flags.DEFINE_string('obs_type', 'tile', 'onehot, tabular, tile for continuous')
flags.DEFINE_string('mdp', './continuous_mdps/obstacle.mdp',
# flags.DEFINE_string('mdp', './mdps/maze.mdp',
                    'File containing the MDP definition (default: mdps/toy.mdp).')
flags.DEFINE_boolean('stochastic', False, 'stochastic transition dynamics or not.')
FLAGS = flags.FLAGS


def main(argv):
    del argv  # Unused.
    mdp_filename = os.path.splitext(os.path.basename(FLAGS.mdp))[0]
    logs = os.path.join(FLAGS.logs, FLAGS.model_class)
    plots = os.path.join(FLAGS.plots, FLAGS.model_class)
    logs = os.path.join(logs, os.path.join(mdp_filename, "stochastic" if FLAGS.stochastic else "deterministic"))
    plots = os.path.join(plots, os.path.join(mdp_filename, "stochastic" if FLAGS.stochastic else "deterministic"))

    folders = [[int(run_mode_folder.split("_")[-1][1:]), run_mode_folder, os.path.join(logs, run_mode_folder)]
                for run_mode_folder in os.listdir(logs)
                if run_mode_folder != ".DS_Store" and
                run_mode_folder != "vanilla" and
                not os.path.isfile(os.path.join(logs, run_mode_folder))]
    n = len(folders)
    color = plt.cm.Blues(np.linspace(0.5, 0.9, n)[::-1])  # This returns RGBA; convert:
    hexcolor = map(lambda rgb: '#%02x%02x%02x' % (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)),
                   tuple(color[:, 0:-1]))
    color = hexcolor #plt.cm.viridis(np.linspace(0, 1, n))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

    folders.sort(reverse=True)
    folders.append([0, "vanilla", os.path.join(logs, "vanilla")])
    for i in range(len(folders)):
        _, run_mode_folder, folder_path = folders[i]
        # if run_mode_folder == ".DS_Store":
        #     continue
        # folder_path = os.path.join(logs, run_mode_folder)
        # if os.path.isfile(folder_path):
        #     continue
        plot_tensorflow_log(folder_path, run_mode_folder)

    plt.xlabel("Episode count")
    plt.ylabel("MSVE")
    plt.legend(loc='upper right', frameon=True)
    # plt.show()
    if not os.path.exists(plots):
        os.makedirs(plots)

    plt.savefig(os.path.join(plots, "msve.png"))


def plot_tensorflow_log(path, run_mode):
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 200,
        'histograms': 1,
        'tensors': 200,
    }
    logs = os.path.join(os.path.join(path, "summaries"), "seed_42")
    filename = os.listdir(logs)[0]
    filepath = os.path.join(logs, filename)
    event_acc = EventAccumulator(filepath, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())
    if not 'train/msve' in event_acc.Tags()["tensors"]:
        return

    msve = event_acc.Tensors('train/msve')

    # steps = len(msve)
    # x = np.arange(steps)
    x = [m[1] for m in msve]
    y = [tf.make_ndarray(m[2]) for m in msve]

    if run_mode == "vanilla":
        plt.plot(x, y, label=run_mode, c="r", alpha=1, linestyle=':')#, marker='v')
    else:
        plt.plot(x, y, label=run_mode, alpha=1, linestyle='-')#, marker='v')

if __name__ == '__main__':
    app.run(main)
