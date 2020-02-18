import os
from absl import app
from absl import flags
from jax import random as jrandom
import network

from utils import *
import experiment
import agents
import utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
np.set_printoptions(precision=3, suppress=1)
plt.style.use('seaborn-notebook')
plt.style.use('seaborn-whitegrid')
from matplotlib.pyplot import cm
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

    properties["model_class"] = filename_list[0]
    properties["mdp"] = filename_list[1]
    i = 2
    if filename_list[i] != "stochastic" and filename_list[i] != "deterministic":
        properties["mdp"] += filename_list[i]
        i += 1
    properties["stochastic"] = filename_list[i]
    i += 1
    properties["env_size"] = filename_list[i]
    i += 1
    if filename_list[i].startswith("lr"):
        properties["lr"] = filename_list[i]
        properties["lrm"] = filename_list[i+1]
        i += 2
    properties["run_mode"] = filename_list[i]
    i += 1
    while filename_list[i] != "summaries":
        properties["run_mode"] += "_" + filename_list[i]
        i += 1
    properties["seed"] = filename_list[i+2]
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

            def movingaverage(values, window):
                weights = np.repeat(1.0, window) / window
                sma = np.convolve(values, weights, 'valid')
                return sma

            x_axis = np.array([np.array(element[0]) for element in dataset][:200])
            y_axis = np.array([np.array(element[1]) for element in dataset][:200])

            y_axis_smooth = movingaverage(y_axis, 10)
            y_axis_smooth = np.concatenate([y_axis[:9], y_axis_smooth])

            plot.plot(x_axis, y_axis, COLORS[properties["run_mode"]], alpha=0.1, linestyle='-')
            plot.plot(x_axis, y_axis_smooth, COLORS[properties["run_mode"]], label=run_mode, alpha=0.5, linestyle='-')
            plot.yaxis.set_minor_formatter(NullFormatter())
            plot.set_ylabel('Episode Steps')
            plot.set_xlabel('Episode Count')

        plot.set_title(title)
        plot.legend()
        fig.add_subplot(plot)
        plt.savefig(folder_properties["plot_filepath"])


if __name__ == '__main__':
    app.run(main)