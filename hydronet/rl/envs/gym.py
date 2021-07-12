"""Environments listed as OpenAI gyms"""
from typing import List, Tuple

import networkx as nx
import logging
import gym

from .actions import WaterClusterActions
from .spaces import AllValidClusters
from .rewards import RewardFunction, BondCountReward

logger = logging.getLogger(__name__)


class WaterCluster(gym.Env):
    """Defines an environment which represents a water cluster as a graph

    The step operation takes a tuple defining a new hydrogen bond: donor ID, and acceptor ID.
    Following the GCPN approach of `You et al. <https://arxiv.org/abs/1806.02473>` we represent the
    state as the bonding graph plus a separate water cluster than can be added to the cluster by adding its step.
    """

    def __init__(self, reward: RewardFunction = None, maximum_size: int = 10,
                 init_cluster: nx.DiGraph = None, record_path: bool = False):
        """Initializes the parameters for the MDP.

        Internal state will be stored as SMILES strings, but but the environment will
        return the new state as an ML-ready fingerprint

        Args:
            reward (RewardFunction): Definition of the reward function
            init_cluster (nx.Graph): Initial molecule as a networkx graph. If None, a single water will be created
            record_path (bool): Whether to record the steps internally.
        """

        super().__init__()
        # Capture the user settings
        if init_cluster is None:
            init_cluster = nx.DiGraph()
            init_cluster.add_node(0)
        if reward is None:
            reward = BondCountReward()
        self.reward_fn = reward
        self.action_space = WaterClusterActions()
        self.init_cluster = init_cluster
        self.record_path = record_path
        self.observation_space = AllValidClusters()
        self.maximum_size = maximum_size

        # Define the state variables
        self._state: nx.DiGraph = self.init_cluster.copy()
        self._path: List[nx.DiGraph] = [self._state.copy()]
        self._counter = 0

        # Ready the environment
        self.reset()

    @property
    def num_steps_taken(self):
        return self._counter

    @property
    def state(self) -> nx.Graph:
        """State as a networkx graph"""
        return self._state

    def get_path(self):
        return list(self._path)

    def reset(self, state: nx.Graph = None):
        """Resets the MDP to its initial state.

        Args:
            state: A graph to use
        """
        if state is None:
            self._state = self.init_cluster.copy()
        else:
            self._state = state

        if self.record_path:
            self._path = [self._state]
        self._counter = 0

    def reward(self):
        """Gets the reward for the state.

        A child class can redefine the reward function if reward other than
        zero is desired.

        Returns:
          Float. The reward for the current state.
        """
        if self._state is None:
            return 0
        return self.reward_fn(self._state)

    def step(self, action: Tuple[int, int, bool]):
        # Add a new node if needed
        donor, acceptor, stop = action
        if max(donor, acceptor) == len(self._state):
            self._state.add_node(len(self._state))

        # Update the water cluster
        self._state.add_edge(donor, acceptor, label='donate')
        self._state.add_edge(acceptor, donor, label='accept')

        # Store the new state
        if self.record_path:
            self._path.append(self._state.copy())

        # Update the action space
        self.action_space.update_actions(self._state)

        # Compute the reward and if we are done
        done = len(self._state) > self.maximum_size or stop

        # Compute the fingerprints for the state
        return self._state, self.reward(), done, {}

    def render(self, **kwargs):
        raise NotImplementedError()
