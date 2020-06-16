from collections import namedtuple

import torch
import torch.nn.functional as F

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ModelOptimizer:

    def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device):
        """
        Optimizes model
        :param policy_net: policy net
        :param target_net: target net
        :param optimizer: optimizer
        :param memory: replay memory
        :param batch_size: batch size
        :param gamma: gamma
        :param device: device
        :return:
        """

        # Skip if replay memory is not filled enough
        if len(memory) < batch_size:
            return

        # Get x transitions from replay memory where x is batch size
        transitions = memory.sample(batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        next_state = torch.cat(batch.next_state)
        reward = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        q_values = policy_net(state).gather(1, action)
        # Compute V(s_{t+1}) for all next states.
        next_q_values = torch.zeros(batch_size, device=device)
        next_q_state_values = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_q_values = (next_q_values * gamma) + reward

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the output
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss
