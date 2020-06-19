import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import torch


class PerformancePlotter:

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

    def save_values_plot(directory, total_frames, values, title, xlabel, ylabel):
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
        # Make path if not yet exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        plt = PerformancePlotter.generate_plot(values, title, xlabel, ylabel)

        plt.savefig(fname=directory + "/" + str(title) + "-frame-{:07d}".format(total_frames) + ".png",
                    format="png",
                    metadata={
                        "Title": str(title) + "-frame-{:07d}".format(total_frames),
                        "Author": "Daniel Pleuss, Clemens Voehringer, Patrick Schmidt, Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of " + str(title)
                    })

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
