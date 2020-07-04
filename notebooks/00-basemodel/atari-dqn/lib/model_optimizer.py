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
    def loss_fn(q_loss, inverse_loss_err, forward_loss_err, beta, lambda1):
        """
        Overall loss function to optimize for all 4 modules
    
        Loss function based on calculation in paper
        """
        loss_ = (1 - beta) * inverse_loss_err
        loss_ += beta * forward_loss_err
        loss_ = loss_.sum() / loss_.flatten().shape[0]
        loss = loss_ + lambda1 * q_loss
        return loss

    def ICM(state1, action, state2, inverse_loss, forward_loss,  encoder, forward_model, inverse_model, forward_scale=1., inverse_scale=1e4): #action is an integer [0:11]
        #"""
        #Intrinsic Curiosity Module (ICM): Calculates prediction error for forward and inverse dynamics
    # 
    # The ICM takes a state1, the action that was taken, and the resulting state2 as inputs
    # (from experience replay memory) and uses the forward and inverse models to calculate the prediction error
    # and train the encoder to only pay attention to details in the environment that are controll-able (i.e. it should
    # learn to ignore useless stochasticity in the environment and not encode that).
    # """
        state1_hat = encoder(state1)
        state2_hat = encoder(state2)
        #Forward model prediction error
        state2_hat_pred = forward_model(state1_hat.detach(), action.detach())
        forward_pred_err = forward_scale * forward_loss(state2_hat_pred, \
                        state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        #Inverse model prediction error
        pred_action = inverse_model(state1_hat, state2_hat) #returns softmax over actions
        inverse_pred_err = inverse_scale * inverse_loss(pred_action, \
                                        action.detach().flatten()).unsqueeze(dim=1)
        return forward_pred_err, inverse_pred_err
        
    
    def optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device ,all_model_params, eta, beta, lambda1, inverse_loss, forward_loss,encoder, forward_model, inverse_model ):
        """
        Optimizes model
        :param policy_net: policy net
        :param target_net: target net
        :param optimizer: optimizer
        :param memory: replay memory
        :param batch_size: batch size
        :param gamma: gamma
        :param device: device
        :param all_model_params: all_model_params
        :param eta: eta
        :param beta: beta
        :param lambda1: lambda1
        :param inverse_loss: inverse_loss
        :param forward_loss: forward_loss

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
               #reward = torch.cat(batch.reward)

        ################################
        forward_pred_err, inverse_pred_err = ModelOptimizer.ICM(state, action, next_state, inverse_loss, forward_loss, encoder, forward_model, inverse_model ) #internal curiosity module
        i_reward = (1. / eta) * forward_pred_err
        i_reward = i_reward.detach()
        reward = torch.transpose(i_reward,0,1)+torch.cat(batch.reward)
        #####################################
        
        # Compute Q(s_t, a)
        q_values = policy_net(state).gather(1, action)
        # Compute V(s_{t+1}) for all next states.
        next_q_values = torch.zeros(batch_size, device=device)
        #next_q_state_values = target_net(non_final_next_states).max(1)[0].detach()
        next_q_values[non_final_mask]= target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_q_values = (next_q_values * gamma) + reward

        # Compute Huber loss
        q_loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        #######################################
        loss = ModelOptimizer.loss_fn(q_loss, forward_pred_err, inverse_pred_err, beta, lambda1)
        #######################################

        # Optimize the output
        optimizer.zero_grad()
        loss.backward()
        for param in all_model_params:
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss

    
  