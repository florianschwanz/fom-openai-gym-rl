import numpy as np
import torch
import torch.autograd as autograd


class ModelOptimizerCuriosity():

    def compute_td_loss(policy_net, target_net, optimizer, memory, batch_size, gamma, num_atoms, vmin, vmax,
                        all_model_params, n_actions, eta, beta, lambda1, inverse_loss, forward_loss,encoder, forward_model,
                        inverse_model):

        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if \
            torch.cuda.is_available() else autograd.Variable(*args, **kwargs)

        state, action, reward, next_state, done = memory.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action = Variable(torch.LongTensor(action))
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        # Calculate curiosity reward
        forward_pred_err, inverse_pred_err = ModelOptimizerCuriosity.ICM(state, action, n_actions, next_state, inverse_loss,
                                                                         forward_loss, encoder,
                                                                forward_model, inverse_model)  # internal curiosity module
        i_reward = (1. / eta) * forward_pred_err
        i_reward = i_reward.detach()
        curiosity_reward = i_reward.squeeze() + reward

        proj_dist = ModelOptimizerCuriosity.projection_distribution(next_state, curiosity_reward, done, target_net, gamma, num_atoms, vmin, vmax)

        dist = policy_net(state)

        # Calculate Q-loss
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        q_loss = -(Variable(proj_dist) * dist.log()).sum(1)
        q_loss = q_loss.mean()

        # Calculate overall loss
        loss = ModelOptimizerCuriosity.loss_fn(q_loss, forward_pred_err, inverse_pred_err, beta, lambda1)


        optimizer.zero_grad()
        loss.backward()
        for param in all_model_params:
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        policy_net.reset_noise()
        target_net.reset_noise()

        return loss

    def projection_distribution(next_state, rewards, dones, target_net, gamma, num_atoms, vmin, vmax):
        batch_size = next_state.size(0)

        delta_z = float(vmax - vmin) / (num_atoms - 1)
        support = torch.linspace(vmin, vmax, num_atoms)

        next_dist = target_net(next_state).data.cpu() * support

        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * gamma * support
        Tz = Tz.clamp(min=vmin, max=vmax)
        b = (Tz - vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

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

    def ICM(state1, action, n_actions, state2, inverse_loss, forward_loss, encoder, forward_model, inverse_model, forward_scale=1.,
            inverse_scale=1e4):  # action is an integer [0:11]
        # """
        # Intrinsic Curiosity Module (ICM): Calculates prediction error for forward and inverse dynamics
        #
        # The ICM takes a state1, the action that was taken, and the resulting state2 as inputs
        # (from experience replay memory) and uses the forward and inverse models to calculate the prediction error
        # and train the encoder to only pay attention to details in the environment that are controll-able (i.e. it should
        # learn to ignore useless stochasticity in the environment and not encode that).
        # """
        state1_hat = encoder(state1)
        state2_hat = encoder(state2)
        # Forward model prediction error
        state2_hat_pred = forward_model(state1_hat.detach(), action.detach(), n_actions)
        forward_pred_err = forward_scale * forward_loss(state2_hat_pred, \
                                                        state2_hat.detach()).sum(dim=1).unsqueeze(dim=1)
        # Inverse model prediction error
        pred_action = inverse_model(state1_hat, state2_hat)  # returns softmax over actions
        inverse_pred_err = inverse_scale * inverse_loss(pred_action, \
                                                        action.detach().flatten()).unsqueeze(dim=1)
        return forward_pred_err, inverse_pred_err