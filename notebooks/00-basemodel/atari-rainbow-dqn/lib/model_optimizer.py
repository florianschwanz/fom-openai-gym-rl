import numpy as np
import torch
import torch.autograd as autograd


class ModelOptimizer():

    def compute_td_loss(policy_net, target_net, optimizer, memory, batch_size, gamma, num_atoms, vmin, vmax):
        Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if \
            torch.cuda.is_available() else autograd.Variable(*args, **kwargs)

        state, action, reward, next_state, done = memory.sample(batch_size)

        state = Variable(torch.FloatTensor(np.float32(state)))
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action = Variable(torch.LongTensor(action))
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        proj_dist = projection_distribution(next_state, reward, done, target_net, gamma, num_atoms, vmin, vmax)

        dist = policy_net(state)

        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist) * dist.log()).sum(1)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
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
