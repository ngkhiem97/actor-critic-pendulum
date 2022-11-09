from learning.models import *

NUM_TILES = 8
NUM_ACTIONS = 1
ACTION_HIGH = 2
ACTION_LOW = -2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = Actor(NUM_TILES, NUM_ACTIONS, ACTION_LOW, ACTION_HIGH)
input = torch.randn(NUM_TILES)
output = actor.sample(input)
print("actor output: {}".format(output))

critic = Critic(NUM_TILES)
input = torch.randn(NUM_TILES)
output = critic(input)
print("critic output: {}".format(output))