import torch
import torch.nn as nn
import torch.nn.functional as F

# Value of y in Phi
PHI_Y = 1152

class Phi(nn.Module):  # (raw state) -> low dim state , encoder
    def __init__(self, input_shape):
        super(Phi, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1)

    def forward(self, x):
        x = F.normalize(x)
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = F.elu(self.conv3(y))
        y = F.elu(self.conv4(y))  # size [1, 32, 3, 3] batch, channels, 3 x 3
        y = y.flatten(start_dim=1)  # size N, 1152
        return y


class Gnet(nn.Module):  # Inverse model: (phi_state1, phi_state2) -> action
    def __init__(self):
        super(Gnet, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.linear1 = nn.Linear(PHI_Y * 2, 256)  # 2 * 1152 = twice the value of y in Phi
        self.linear2 = nn.Linear(256, 4)  # Action number dynamisch berücksichtigen

    def forward(self, state1, state2):
        x = torch.cat((state1, state2), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        y = F.softmax(y, dim=1)
        return y


class Fnet(nn.Module):
    def __init__(self, num_actions):
        super(Fnet, self).__init__()
        # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.linear1 = nn.Linear(PHI_Y + num_actions, 256)  # Anpassung notendig! Action number dynamisch berücksichtigen
        self.linear2 = nn.Linear(256, PHI_Y)  # Anpassung notendig! Action number dynamisch berücksichtigen

    def forward(self, state, action, num_actions):
        action_ = torch.zeros(action.shape[0], num_actions)  # Anpassung notendig!
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze().cpu()), dim=0)

        indices = indices.tolist()
        action_[indices] = 1.
        x = torch.cat((state, action_), dim=1)
        y = F.relu(self.linear1(x))
        y = self.linear2(y)
        return y