from __future__ import print_function
import numpy as np
from coder.tile import PendulumTileCoder
from learning.utils import *
from learning.models import Actor, Critic
import torch
from collections import deque
from gym import Env
import random
from torch.distributions import Normal
import torch.nn.functional as F

ACTOR_LR = 0.0001
CRITIC_LR = 0.0002
MEMORY_SIZE = 10000
OBSERVATIONS = 2
EPOCHS = 1000
STEPS_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99

class ActorCriticAgent(): 
    def __init__(self, gamma=GAMMA):
        self.rand_generator = None

        # define step-size for actor, critic, and average reward
        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        # define the tile coder
        self.tc = None

        # define the averagereward, actor weights, and critic weights
        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        # define the actions
        self.actions = None

        # define the softmax probability, previous tiles, and last action
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

        # gamma
        self.gamma = gamma
    
    def init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "action_high": float,
            "action_low": float,
            "seed": int
        }
        """

        # set random seed for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed")) 

        # initialize the tile coder
        self.iht_size = agent_info.get("iht_size")
        self.num_tilings = agent_info.get("num_tilings")
        self.num_tiles = agent_info.get("num_tiles")
        self.tc = PendulumTileCoder(iht_size=self.iht_size, num_tilings=self.num_tilings, num_tiles=self.num_tiles)

        # set step-size accordingly
        self.actor_step_size = agent_info.get("actor_step_size")/self.num_tilings
        self.critic_step_size = agent_info.get("critic_step_size")/self.num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        # get number of actions
        self.actions = list(range(agent_info.get("num_actions")))
        self.action_high = agent_info.get("action_high")
        self.action_low = agent_info.get("action_low")

        # Set initial values of average reward, actor weights, and critic weights
        self.actor = Actor(self.num_tilings, agent_info.get("num_actions"), self.action_high, self.action_low)
        self.critic = Critic(self.num_tilings)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.avg_reward = 0.0

        # Saved values
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None

        # define the memory
        self.memory = deque()
    
    def act(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        tensor_active_tiles = torch.tensor(active_tiles, dtype=torch.float32)
        with torch.no_grad():
            return self.actor.sample(tensor_active_tiles)

    def start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """
        x, y, ang_vel = state
        active_tiles = self.tc.get_tiles(x, y, ang_vel)
        current_action = self.act(active_tiles)
        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)
        return self.last_action

    def learn(self, env: Env, epochs=EPOCHS, steps_size=STEPS_SIZE):
        """Run the algorithm for a number of episodes.
        Args:
            observations (int): number of observations to run the algorithm for
            epochs (int): number of epochs to run the algorithm for
        """
        steps = 0
        for epoch in range(EPOCHS):
            state = env.reset()
            self.start(state)
            episode_reward = 0
            k = 0
            for time_steps in range(steps_size):
                env.render()
                x, y, ang_vel = state
                actile_tiles = self.tc.get_tiles(x, y, ang_vel)
                action = self.act(actile_tiles)
                next_state, reward, done, _ = env.step([action])
                episode_reward += reward
                reward = (reward + 8.1) / 8.1
                self.remember(state, next_state, action, reward, done)
                state = next_state
                if k == 32 or time_steps == steps_size-1:
                    k = 0
                    steps += 1
                    experiences = self.sample()
                    batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*experiences)
                    batch_state = torch.FloatTensor(batch_state)
                    batch_next_state = torch.FloatTensor(batch_next_state)
                    batch_action = torch.FloatTensor(batch_action).unsqueeze(1)
                    batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1)
                    batch_done = torch.FloatTensor(batch_done).unsqueeze(1)
                    with torch.no_grad():
                        value_target = batch_reward + self.gamma * (1 - batch_done) * self.critic(batch_next_state)
                        advantage = value_target - self.critic(batch_state)
                    mu, std = self.actor(batch_state)
                    n = Normal(mu, std)
                    log_prob = n.log_prob(batch_action)
                    actor_loss = - log_prob * advantage
                    actor_loss = actor_loss.mean()
                    self.actor_optim.zero_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    critic_loss = F.mse_loss(value_target, self.critic(batch_state))
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
                    self.memory.clear()
                if done:
                    break
                state = next_state

            print(f"Epoch: {epoch}, Reward: {episode_reward}")
    def message(self, message):
        if message == 'get avg reward':
            return self.avg_reward

    def remember(self, state, next_state, action, reward, done):
        active_tiles = self.tc.get_tiles(state[0], state[1], state[2])
        next_active_tiles = self.tc.get_tiles(next_state[0], next_state[1], next_state[2])
        self.memory.append((active_tiles, next_active_tiles, action, reward, done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)