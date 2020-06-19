import math
import random

import numpy as np
import torch
import torch.autograd as autograd

steps_done = 0


class ActionSelector:

    def select_action(state, n_actions, total_frames, policy_net, epsilon_end, epsilon_start, epsilon_decay,
                      vmin, vmax, num_atoms, device, USE_CUDA):
        """
        Selects an action based on the current state
        :param state: state of the environment
        :param n_actions: number of possible actions
        :param total_frames: number of frames since the beginning
        :param policy_net: policy net
        :param epsilon_end: epsilon end
        :param epsilon_start: epsilon start
        :param epsilon_decay: epsilon decay
        :param device: device
        :return:
        """
        sample = random.random()

        # Epsilon threshold is decreasing over time
        # at the beginning more actions are done randomly
        # later the amount of random actions tends to 0
        eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
                        math.exp(-1. * total_frames / epsilon_decay)
        if sample > eps_threshold:
            # Select action based on policy net
            with torch.no_grad():
                Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if \
                    USE_CUDA else autograd.Variable(*args, **kwargs)

                with torch.no_grad():
                    state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0))
                    dist = policy_net.forward(state).data.cpu()
                    dist = dist * torch.linspace(vmin, vmax, num_atoms)
                    action = dist.sum(2).max(1)[1].numpy()[0]
                return action
        else:
            # Select random action
            return random.randrange(n_actions)
