import os

import numpy as np


class PerformanceLogger:

    def log_episode(directory, total_episodes, total_frames, total_duration, total_original_rewards,
                    total_shaped_rewards, episode_frames, episode_original_reward,
                    episode_shaped_reward, episode_loss, episode_duration):

        # Make path if not yet exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        avg_frames_per_minute = total_frames / (total_duration / 60)
        # avg_episodes_per_minute = total_episodes / (total_duration / 60)
        avg_original_reward_per_episode = np.mean(total_original_rewards[-50:])
        avg_shaped_reward_per_episode = np.mean(total_shaped_rewards[-50:])

        line = ("{: 5d}".format(total_episodes)
                + " {: 5d}".format(episode_frames) + "f"
                + " {: 4d}".format(round(episode_duration)) + "s"
                + " {: 4d}".format(round(avg_frames_per_minute)) + "f/min"
                + "     "
                + " reward {: 5f}".format(round(episode_original_reward, 2))
                + " reward(shaped) {: 5f}".format(round(episode_shaped_reward, 2))
                + " avg reward per episode {: 3f}".format(round(avg_original_reward_per_episode, 2))
                + " avg reward(shaped) per episode {: 3f}".format(round(avg_shaped_reward_per_episode, 2))
                + " loss " + str(round(episode_loss, 4)))

        # Print log
        print(line)

        # Write log into file
        log_file = open(directory + "/log.txt", "a")
        log_file.write(line)
        log_file.close()
