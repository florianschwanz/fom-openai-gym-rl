import os
import glob
from email.utils import formatdate

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


class ScreenPlotter:

    # Maximum number of files we want to store
    MAX_FILES = 3

    def display_screen_plot(total_frames, env, title, device):
        """
         Plots current state of the screen
        :param total_frames: number of frames since the beginning
        :param env: environment to extract screen from
        :param title: title
        :param device: device
        :return:
        """

        plt = ScreenPlotter.generate_plot(env, title, device)

        plt.show()
        plt.close()

    def save_screen_plot(output_directory, run_directory, total_frames, env, name, title, device, prune=True):
        """
        Saves current state of the screen as png file
        :param directory to save plot in
        :param total_frames: number of frames since the beginning
        :param env: environment to extract screen from
        :param title: title
        :param device: device
        :return:
        """

        target_directory = output_directory + "/" +run_directory
        symlink_directory = "latest"

        # Make path if not yet exists
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if not os.path.exists(target_directory):
            os.mkdir(target_directory)

        # Create symlink
        if os.path.islink(symlink_directory):
            os.unlink(symlink_directory)
        os.symlink(target_directory, symlink_directory)

        plt = ScreenPlotter.generate_plot(env, title, device)

        plt.savefig(fname=target_directory + "/" + str(name) + "-frame-{:07d}".format(total_frames) + ".png",
                    format="png",
                    metadata={
                        "Title": str(title) + "-frame-{:07d}".format(total_frames),
                        "Author": "Daniel Pleus, Clemens Voehringer, Patrick Schmidt, Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of " + str(title)
                    })

        # Close plot
        plt.close()

        if prune:
            ScreenPlotter.prune_storage(target_directory, str(title) + "*.png")

    def generate_plot(env, title, device):
        """
        Generates plot of a screen
        :param env: environment to extract screen from
        :param title: title
        :param device: device
        :return: plot
        """

        resize = T.Compose([T.ToPILImage(),
                            # T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

        screen = env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        screen = resize(screen).unsqueeze(0).to(device)

        plt.figure()
        plt.imshow(screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title(title)
        return plt

    def prune_storage(prune_directory, pattern):
        list_of_files = glob.glob(prune_directory + "/" + pattern)

        while len(list_of_files) > ScreenPlotter.MAX_FILES:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(prune_directory + "/" + pattern)