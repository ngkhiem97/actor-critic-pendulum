from coder.tile import PendulumTileCoder
import numpy as np
import itertools

# define angle and angular velocity ranges
x = np.linspace(-1, 1, 2)
y = np.linspace(-1, 1, 2)
vels = np.linspace(-8, 8, 2)
test_obs = list(itertools.product(x, y, vels))

tile_coder = PendulumTileCoder(iht_size=4096, num_tilings=8, num_tiles=2)

result=[]
for obs in test_obs:
    x, y, ang_vel = obs
    tiles = tile_coder.get_tiles(x=x, y=y, ang_vel=ang_vel)
    result.append(tiles)

print(result)