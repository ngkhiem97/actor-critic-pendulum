import torch
import torch.nn as nn
import numpy as np

EPSILON = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

# Actor Network
class Actor(torch.nn.Module):
    def __init__(self, num_tiles, action_size, action_high:np.array, action_low:np.array):
        super(Actor, self).__init__()
        # input layer
        self.fc1 = nn.Linear(num_tiles, 64)

        # hidden layer
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        # output layer
        self.fc_mean = nn.Linear(256, action_size)
        self.fc_std = nn.Linear(256, action_size)
        self.mean = nn.Tanh()
        self.std = nn.Softplus()

        # initialize weights
        self.apply(weights_init_)

        # set action range
        self.action_high = action_high
        self.action_low = action_low

    def forward(self, active_tiles:torch.Tensor):
        x = active_tiles
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean(self.fc_mean(x)) * self.action_high
        std = self.std(self.fc_std(x)) + EPSILON
        return mean, std

    def sample(self, active_tiles:torch.Tensor):
        mean, std = self.forward(active_tiles)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        return np.clip(action.item(), self.action_low, self.action_high)

# Critic Network
class Critic(torch.nn.Module):
    def __init__(self, num_tiles):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(num_tiles, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, active_tiles:torch.Tensor):
        x = active_tiles
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x