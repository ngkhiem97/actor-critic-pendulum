import torch

class Actor(torch.nn.Module):
    def __init__(self, iht_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(iht_size, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 256)
        self.fc4 = torch.nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=0)
        return x

class Critic(torch.nn.Module):
    def __init__(self, iht_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(iht_size+action_size, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, 256)
        self.fc4 = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x