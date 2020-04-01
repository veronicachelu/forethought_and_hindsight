from absl import app
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from utils import *

plt.style.use('ggplot')
np.set_printoptions(precision=3, suppress=1)
plt.style.use('seaborn-notebook')
plt.style.use('seaborn-whitegrid')
from matplotlib.ticker import NullFormatter

flags.DEFINE_string('results', str((os.environ['RESULTS'])), 'where to load results')
flags.DEFINE_string('plots', str((os.environ['PLOTS'])), 'where to save results')
FLAGS = flags.FLAGS

COLORS = {"vanilla": 'r',
          "replay": 'g',
          "dyna": 'b',
          "priority_replay": 'c',
          "priority_dyna": 'm',
          "onpolicy": 'y'}

def parse_filename(filename):
    filename_list = filename.split("-")[1].split("_")
    properties = {}
    i = 0
    properties["stochastic"] = filename_list[i]
    i += 1
    properties["run_mode"] = filename_list[i]
    i += 1
    while filename_list[i] != "summaries":
        properties["run_mode"] += "_" + filename_list[i]
        i += 1
    return properties

def parse_folder(folder):
    filename_list = folder.split("_")
    properties = {}
    properties["model_class"] = filename_list[1]
    properties["mdp"] = filename_list[0]
    properties["stochastic"] = filename_list[2]
    return properties

def main(argv):
    del argv  # Unused.
    for folder in os.listdir(FLAGS.results):
        if folder == ".DS_Store":
            continue
        fig = plt.figure(figsize=(6, 4))
        outer = gridspec.GridSpec(1, 1, wspace=0.02, hspace=0.05)
        plot = plt.Subplot(fig, outer[0])
        folder_properties = parse_folder(folder)
        title = "{}_{}_{}".format(folder_properties["mdp"],
                                     folder_properties["model_class"],
                                     folder_properties["stochastic"])

        folder_properties["folder_path"] = os.path.join(FLAGS.results, folder)

        folder_properties["plots_path"] = os.path.join(FLAGS.plots, folder)
        folder_properties["plot_filepath"] = os.path.join(folder_properties["plots_path"], "plot.png")

        if not os.path.exists(folder_properties["plots_path"]):
            os.makedirs(folder_properties["plots_path"])

        for file in os.listdir(folder_properties["folder_path"]):
            if file == ".DS_Store":
                continue
            properties = parse_filename(file)
            run_mode = properties["run_mode"]
            if "lr" in properties.keys() and "lrm" in properties.keys():
                run_mode += "_{}_{}".format(properties["lr"], properties["lrm"])
            properties["filepath"] = os.path.join(folder_properties["folder_path"], file)
            # properties["plot_filepath"] = os.path.join(folder_properties["plots_path"], file)
            # run - tabular_maze_stochastic_1x_dyna_summaries_seed_42 - tag - test_num_steps.csv

            dataset = tf.data.experimental.CsvDataset(
                properties["filepath"],
                [tf.int32, # Required field, use dtype or empty tensor
                 tf.float32,  # Required field, use dtype or empty tensor
                 ],
                header=True,
                select_cols=[1, 2] , # Only parse last two columns
            )

            def smooth(scalars, weight=0.6):  # Weight between 0 and 1
                last = scalars[0]  # First value in the plot (first timestep)
                smoothed = list()
                for point in scalars:
                    smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                    smoothed.append(smoothed_val)  # Save it
                    last = smoothed_val  # Anchor the last smoothed value

                return smoothed

            def movingaverage(values, window):
                weights = np.repeat(1.0, window) / window
                sma = np.convolve(values, weights, 'valid')
                return sma

            x_axis = np.array([np.array(element[0]) for element in dataset][:200])
            y_axis = np.array([np.array(element[1]) for element in dataset][:200])

            y_axis_smooth = smooth(y_axis)
            # y_axis_smooth = np.concatenate([y_axis[:9], y_axis_smooth])

            # plot.plot(x_axis, y_axis, COLORS[properties["run_mode"]], alpha=0.5, linestyle='-')
            plot.plot(x_axis, y_axis_smooth, COLORS[properties["run_mode"]], label=run_mode, alpha=1.0, linestyle='-')
            plot.yaxis.set_minor_formatter(NullFormatter())
            plot.set_ylabel('Cumulative Reward')
            plot.set_xlabel('Steps')

        plot.grid(0)

        plot.set_title(title)
        plot.legend()
        fig.add_subplot(plot)
        plt.savefig(folder_properties["plot_filepath"])


if __name__ == '__main__':
    app.run(main)