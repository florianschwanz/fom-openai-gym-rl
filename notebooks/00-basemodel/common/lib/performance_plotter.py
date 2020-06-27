import glob
import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import torch


class PerformancePlotter:

    # Maximum number of files we want to store
    MAX_FILES = 3

    def display_values_plot(values, title, xlabel, ylabel):
        """
        Plots values to screen
        :param values: values to be plotted
        :param title:  title
        :param xlabel: label of x-axis
        :param ylabel: label of y-axis
        :return:
        """

        plt = PerformancePlotter.generate_plot(values, title, xlabel, ylabel)

        plt.show()

    def save_values_plot(output_directory, run_directory, total_frames, values, title, xlabel, ylabel):
        """
        Saves plot to png file
        :param directory to save plot in
        :param total_frames: number of frames since the beginning
        :param values: values to be plotted
        :param title:  title
        :param xlabel: label of x-axis
        :param ylabel: label of y-axis
        :return:
        """

        target_directory = PerformancePlotter.prepare_directory(output_directory, run_directory)

        plt = PerformancePlotter.generate_plot(values, title, xlabel, ylabel)

        plt.savefig(fname=target_directory + "/" + str(title).replace(" ", "-") + "-frame-{:07d}".format(total_frames) + ".png",
                    format="png",
                    metadata={
                        "Title": str(title) + "-frame-{:07d}".format(total_frames),
                        "Author": "Daniel Pleus, Clemens Voehringer, Patrick Schmidt, Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of " + str(title)
                    })

        plt.close()

        PerformancePlotter.prune_storage(target_directory, str(title).replace(" ", "-") + "*.png")

    def generate_plot(values, title, xlabel, ylabel):
        """
        Generates plot of values
        :param title:  title
        :param xlabel: label of x-axis
        :param ylabel: label of y-axis
        :return:
        """
        plt.figure(2)
        plt.clf()
        durations_t = torch.tensor(values, dtype=torch.float)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        return plt

    def prune_storage(prune_directory, pattern):
        list_of_files = glob.glob(prune_directory + "/" + pattern)

        while len(list_of_files) > PerformancePlotter.MAX_FILES:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(prune_directory + "/" + pattern)

    def prepare_directory(output_directory, run_directory):
        target_directory = output_directory + "/" + run_directory
        symlink_directory = "latest"

        # Make path if not yet exists
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if not os.path.exists(target_directory):
            os.mkdir(target_directory)

        # Create symlink
        if os.path.islink(output_directory + "/" + symlink_directory):
            os.unlink(output_directory + "/" + symlink_directory)
        os.symlink(run_directory, output_directory + "/" + symlink_directory)

        return target_directory
