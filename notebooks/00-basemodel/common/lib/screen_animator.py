import glob
import os

import imageio
from tqdm import tqdm


class ScreenAnimator:
    # Maximum number of files we want to store
    MAX_FILES = 3

    def save_screen_animation(directory, total_episodes, title, prune=True):
        list_of_screenshots = glob.glob(directory + "/gif-screenshot*.png")

        # Render gif
        images = []
        # progress_bar_render = tqdm(sorted(list_of_screenshots), unit='frames', desc="Render gif")
        for filename in sorted(list_of_screenshots):
            images.append(imageio.imread(filename))
        imageio.mimsave(directory + "/" + str(title) + "-episode-{:07d}".format(total_episodes) + ".gif", images)

        # Remove screenshots
        for filename in sorted(list_of_screenshots):
            os.remove(filename)

        if prune:
            ScreenAnimator.prune_storage(directory, str(title) + "*.gif")

    def prune_storage(directory, pattern):
        list_of_files = glob.glob(directory + "/" + pattern)

        while len(list_of_files) > ScreenAnimator.MAX_FILES:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(directory + "/" + pattern)
