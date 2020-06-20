from argument_extractor import ArgumentExtractor


class PotentialBasedRewardShaper:

    def reward(self, **kwargs):
        """
        Gives an additional reward relative to recent episode rewards
        :return: shaped reward
        """

        original_reward = ArgumentExtractor.extract_argument(kwargs, "reward", 0)
        current_episode_reward = ArgumentExtractor.extract_argument(kwargs, "current_episode_reward", 0)
        max_episode_reward = ArgumentExtractor.extract_argument(kwargs, "max_episode_reward", 0)
        min_episode_reward = ArgumentExtractor.extract_argument(kwargs, "min_episode_reward", 0)

        if original_reward != 0 \
                and max_episode_reward != None \
                and min_episode_reward != None \
                and min_episode_reward != max_episode_reward:
            return 1 + ((current_episode_reward - max_episode_reward) / (max_episode_reward - min_episode_reward))
        else:
            return 0
