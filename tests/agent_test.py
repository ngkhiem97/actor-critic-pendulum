from learning.agent import ActorCriticAgent
import numpy as np

agent_info = {
    "iht_size": 4096,
    "num_tilings": 8,
    "num_tiles": 8,
    "actor_step_size": 1e-1,
    "critic_step_size": 1e-0,
    "avg_reward_step_size": 1e-2,
    "num_actions": 1,
    "seed": 99,
    "action_high": 2,
    "action_low": -2
}

test_agent = ActorCriticAgent()
test_agent.init(agent_info)

state = [-1, 0., 0.]

test_agent.start(state)

print("agent active_tiles: {}".format(test_agent.prev_tiles))
print("agent selected action: {}".format(test_agent.last_action))