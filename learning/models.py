import torch
import torch.nn as nn

EPSILON = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)

# Actor Network
class Actor(torch.nn.Module):
    def __init__(self, num_tiles, action_size, action_high, action_low):
        super(Actor, self).__init__()
        # input layer
        self.fc1 = nn.Linear(num_tiles, 64)

        # hidden layer
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)

        # output layer
        self.mean = nn.Linear(256, action_size)
        self.log_std = nn.Linear(256, action_size)

        # initialize weights
        self.apply(weights_init_)

        # define scale and bias
        self.action_scale = torch.FloatTensor((action_high - action_low) / 2.)
        self.action_bias = torch.FloatTensor((action_high + action_low) / 2.)

    def forward(self, active_tiles:torch.Tensor):
        x = active_tiles.view(active_tiles.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

    def sample(self, active_tiles:torch.Tensor):
        mean, log_std = self.forward(active_tiles)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + EPSILON)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)

# Critic Network
class Critic(torch.nn.Module):
    def __init__(self, num_tiles, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(num_tiles, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, active_tiles:torch.Tensor, action):
        active_tiles = active_tiles.view(active_tiles.size(0), -1)
        x = torch.cat((active_tiles, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x