import numpy as np

class PerformanceLogger:

    def log_episode(total_episodes, total_frames, total_duration, episode_frames, episode_original_reward,
                    episode_shaped_reward, episode_original_rewards, episode_shaped_rewards, episode_loss, episode_duration):
        avg_frames_per_minute = total_frames / (total_duration / 60)
        avg_episodes_per_minute = total_episodes / (total_duration / 60)
        avg_original_reward_per_episode = np.mean(episode_original_rewards[-50:])
        avg_shaped_reward_per_episode = np.mean(episode_shaped_rewards[-50:])

        print("Episode  " + "{: 5d}".format(total_episodes) + " (" + str(episode_frames)
              + " frames / " + str(round(episode_duration)) + "s)")
        print(" avg episodes per minute " + str(round(avg_episodes_per_minute, 2))
              + " frames per minute " + str(round(avg_frames_per_minute, 2)))
        print(" reward original " + str(round(episode_original_reward, 2))
              + " shaped " + str(round(episode_shaped_reward, 2)))
        print(" avg reward original " + str(round(avg_original_reward_per_episode))
              + " shaped " + str(round(avg_shaped_reward_per_episode)))
        print(" loss " + str(round(episode_loss, 4)))

    def log_episode_short(total_episodes, total_frames, total_duration, total_original_rewards,
                          total_shaped_rewards, episode_frames, episode_original_reward,
                          episode_shaped_reward, episode_loss, episode_duration):
        avg_frames_per_minute = total_frames / (total_duration / 60)
        avg_episodes_per_minute = total_episodes / (total_duration / 60)
        avg_original_reward_per_episode = np.mean(total_original_rewards[-50:])
        avg_shaped_reward_per_episode = np.mean(total_shaped_rewards[-50:])

        print("{: 5d}".format(total_episodes)
              + " {: 5d}".format(episode_frames) + "f"
              + " {: 4d}".format(round(episode_duration)) + "s"
              + " {: 4d}".format(round(avg_frames_per_minute)) + "f/min"
              + "     "
              + " reward {: 5f}".format(round(episode_original_reward, 2))
              + " reward(shaped) {: 5f}".format(round(episode_shaped_reward, 2))
              + " avg reward per episode {: 3f}".format(round(avg_original_reward_per_episode, 2))
              + " avg reward(shaped) per episode {: 3f}".format(round(avg_shaped_reward_per_episode, 2))
              + " loss " + str(round(episode_loss, 4)))
