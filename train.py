import gym
import coder

env = gym.make('Pendulum-v1')
env.reset()

ndims_state = env.observation_space.shape[0] # 3
ndim_action = env.action_space.shape[0]      # 1

while True:
    env.render()
    action = env.action_space.sample()
    env.step(action)