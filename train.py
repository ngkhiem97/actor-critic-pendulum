import gym
import coder

env = gym.make('Pendulum-v1')
env.reset()
state_space = env.observation_space
print("state_space: {}".format(state_space))
action_space = env.action_space
print("action_space: {}".format(action_space))
sample_state = state_space.sample()
print("sample_state: {}".format(sample_state))
sample_action = action_space.sample()
print("sample_action: {}".format(sample_action))
while True:
    env.render()
    action = env.action_space.sample()
    env.step(action)