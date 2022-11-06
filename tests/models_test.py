from learning.models import *

actor = Actor(4096, 2)
input = torch.randn(4096)
output = actor(input)
print("actor output: {}".format(output))

critic = Critic(4096, 3)
input = torch.randn(4096+3)
output = critic(input)
print("critic output: {}".format(output))