import glob
import os

import imageio
from tqdm import tqdm


class ScreenAnimator:
    # Maximum number of files we want to store
    MAX_FILES = 3

    def save_screen_animation(output_directory, run_directory, total_episodes, title, prune=True):
        target_directory = ScreenAnimator.prepare_directory(output_directory, run_directory)

        list_of_screenshots = glob.glob(target_directory + "/gif-screenshot*.png")

        # Render gif
        images = []
        # progress_bar_render = tqdm(sorted(list_of_screenshots), unit='frames', desc="Render gif")
        for filename in sorted(list_of_screenshots):
            images.append(imageio.imread(filename))
        imageio.mimsave(target_directory + "/" + str(title) + "-episode-{:07d}".format(total_episodes) + ".gif", images)

        # Remove screenshots
        for filename in sorted(list_of_screenshots):
            os.remove(filename)

        if prune:
            ScreenAnimator.prune_storage(target_directory, str(title) + "*.gif")

    def prune_storage(prune_directory, pattern):
        list_of_files = glob.glob(prune_directory + "/" + pattern)

        while len(list_of_files) > ScreenAnimator.MAX_FILES:
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
