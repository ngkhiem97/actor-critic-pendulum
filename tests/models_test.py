from learning.models import *

actor = Actor(4096, 3, 256)
input = torch.zeros(4096)
output = actor(input)
print(output)