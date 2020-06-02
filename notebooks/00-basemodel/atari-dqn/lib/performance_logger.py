class PerformanceLogger:

    def log_episode(total_episodes, total_frames, total_duration, episode_frames, episode_original_reward,
                    episode_shaped_reward, episode_loss, episode_duration):
        avg_frames_per_minute = total_frames / (total_duration / 60)
        avg_episodes_per_minute = total_episodes / (total_duration / 60)

        print("Episode  " + "{: 5d}".format(total_episodes) + " (" + str(episode_frames)
              + " frames / " + str(round(episode_duration)) + "s)")
        print(" avg episodes per minute " + str(round(avg_episodes_per_minute, 2))
              + " frames per minute " + str(round(avg_frames_per_minute, 2)))
        print(" reward original " + str(round(episode_original_reward, 2))
              + " shaped " + str(round(episode_shaped_reward, 2)))
        print(" loss " + str(round(episode_loss, 4)))

    def log_episode_short(total_episodes, total_frames, total_duration, episode_frames, episode_original_reward,
                          episode_shaped_reward, episode_loss, episode_duration):
        avg_frames_per_minute = total_frames / (total_duration / 60)
        avg_episodes_per_minute = total_episodes / (total_duration / 60)

        print("{: 5d}".format(total_episodes)
              + "  {: 5d}".format(total_frames) + "f " + "{: 4d}".format(round(episode_duration)) + "s"
              + " " + str(round(avg_frames_per_minute)) + "f/min"
              + "     "
              + " reward " + str(round(episode_original_reward))
              + " reward(shaped) " + str(round(episode_shaped_reward))
              + " loss " + str(round(episode_loss, 4)))
