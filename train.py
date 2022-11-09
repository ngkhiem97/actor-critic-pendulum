import gym
import coder
from learning.agent import ActorCriticAgent

env = gym.make('Pendulum-v1')
num_actions = env.action_space.shape[0]
action_high = env.action_space.high[0]
action_low = env.action_space.low[0]

print("num_actions: {}".format(num_actions))
print("action_high: {}".format(action_high))
print("action_low: {}".format(action_low))

agent_info = {
    "iht_size": 4096,
    "num_tilings": 8,
    "num_tiles": 3,
    "actor_step_size": 1e-1,
    "critic_step_size": 1e-0,
    "avg_reward_step_size": 1e-2,
    "num_actions": num_actions,
    "seed": 99,
    "action_high": action_high,
    "action_low": action_low
}
agent = ActorCriticAgent()
agent.init(agent_info)
agent.learn(env)