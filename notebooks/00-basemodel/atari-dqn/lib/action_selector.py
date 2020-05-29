import math
import random

import torch

steps_done = 0


class ActionSelector:

    def select_action(state, n_actions, policy_net, epsilon_end, epsilon_start, epsilon_decay, device):
        """
        Selects an action based on the current state
        :param state: state of the environment
        :param n_actions: number of possible actions
        :param policy_net: policy net
        :param epsilon_end: epsilon end
        :param epsilon_start: epsilon start
        :param epsilon_decay: epsilon decay
        :param device: device
        :return:
        """
        global steps_done
        sample = random.random()

        # Epsilon threshold is decreasing over time
        # at the beginning more actions are done randomly
        # later the amount of random actions tends to 0
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
                        math.exp(-1. * steps_done / epsilon_decay)
        steps_done += 1
        if sample > eps_threshold:
            # Select action based on policy net
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            # Select random action
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
