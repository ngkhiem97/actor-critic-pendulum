from __future__ import print_function
from abc import ABCMeta, abstractmethod
import numpy as np
from coder.tile import PendulumTileCoder
from learning.utils import *
from learning.models import Actor, Critic
import torch

ACTOR_LR = 0.0001
CRITIC_LR = 0.0002

class BaseAgent:
    """
    Implements the base agent abstract class
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """

class ActorCriticAgent(BaseAgent): 
    def __init__(self):
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
        self.actor = Actor(agent_info.get("num_actions"), self.num_tiles, self.action_high, self.action_low)
        self.critic = Critic(self.num_tiles)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.avg_reward = 0.0

        # Saved values
        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def act(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        tensor_active_tiles = torch.tensor(active_tiles, dtype=torch.float32)
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

    def step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on 
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """
        x, y, ang_vel = state
        active_tiles = self.tc.get_tiles(x, y, ang_vel)

        # Calculate the state value
        tensor_prev_tiles = torch.tensor(self.prev_tiles, dtype=torch.float32)
        tensor_active_tiles = torch.tensor(active_tiles, dtype=torch.float32)
        previous_state_value = self.critic(tensor_prev_tiles)
        current_state_value = self.critic(tensor_active_tiles)

        # Convert value to tensor
        tensor_reward = torch.tensor(reward, dtype=torch.float32)

        # Calculate the TD error
        delta = torch.tensor(reward, dtype=torch.float32) - torch.tensor(self.avg_reward, dtype=torch.float32) + \
                torch.sum(current_state_value, dim=1) - torch.sum(previous_state_value, dim=1)

        # Update average reward
        self.avg_reward += self.avg_reward_step_size * delta

        # Update critic weights
        loss = torch.sum(delta * self.critic_step_size, dim=1)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        # Update actor weights
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        # Update action
        current_action = self.actor(active_tiles)
        self.prev_tiles = active_tiles
        self.last_action = current_action
        return self.last_action


    def message(self, message):
        if message == 'get avg reward':
            return self.avg_reward
