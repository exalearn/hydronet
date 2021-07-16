from typing import Dict, Any
import logging

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import BoundedArraySpec
from tf_agents.typing import types
import tensorflow as tf
import networkx as nx

from hydronet.data import graph_is_valid
from hydronet.importing import create_inputs_from_nx
from hydronet.rl.envs.rewards import RewardFunction, BondCountReward

logger = logging.getLogger(__name__)


class SimpleEnvironment(PyEnvironment):
    """Environment for gradually growing a water cluster. Terminates only when a maximum size has been met

    The state is the state of the graph (i.e., atom types, bond types, connectivity).
    """

    def __init__(self, reward: RewardFunction = None, maximum_size: int = 10,
                 init_cluster: nx.DiGraph = None, record_path: bool = False,
                 discount_factor: float = 1.0):
        super().__init__()
        # Capture the user settings
        if init_cluster is None:
            init_cluster = nx.DiGraph()
            init_cluster.add_node(0, label='O')
            init_cluster.add_node(1, label='O')
            init_cluster.add_edge(0, 1, label='donate')
            init_cluster.add_edge(1, 0, label='accept')
        if reward is None:
            reward = BondCountReward()
        self.reward_fn = reward
        self.init_cluster = init_cluster
        self.record_path = record_path
        self.maximum_size = maximum_size
        self.discount_factor = discount_factor

        # Define the initial state
        self._state = init_cluster.copy()

    def observation_spec(self) -> types.NestedArraySpec:
        # TF-Agents require fixed size arrays. We know the number of bonds cannot be more than 4 times
        #  the number of atoms, as each water can only donate 2 bonds.
        # We use 1 + the maximum number of atoms so that we can include a "ghost" atom ready to be added to the cluster
        n = self.maximum_size + 1
        return {
            'n_atoms': BoundedArraySpec((), minimum=0, dtype='int32'),
            'atom': BoundedArraySpec((n,), minimum=0, maximum=1, dtype='int32'),
            'bond': BoundedArraySpec((n * 4,), minimum=0, maximum=1, dtype='int32'),
            'connectivity': BoundedArraySpec((n * 4, 2), minimum=0, maximum=self.maximum_size, dtype='int32')
        }

    def action_spec(self) -> types.NestedArraySpec:
        return BoundedArraySpec((2,), dtype='int32', minimum=0, maximum=self.maximum_size)

    def get_state_as_tensors(self) -> Dict[str, tf.Tensor]:
        # Get the data as arrays
        simple_graph = create_inputs_from_nx(self._state)

        # Build the buffered arrays
        n = self.maximum_size + 1
        output = {
            'n_atoms': simple_graph['n_atoms'],
            'atom': np.zeros((n,), dtype=np.int32),
            'bond': np.zeros((n * 4,), dtype=np.int32),
            'connectivity': np.zeros((n * 4, 2), dtype=np.int32)
        }
        for i in ['atom', 'bond']:
            output[i][:len(simple_graph[i])] = simple_graph[i]
        output['connectivity'][:len(simple_graph['connectivity']), :] = np.array(simple_graph['connectivity'])

        return dict((k, tf.convert_to_tensor(v)) for k, v in output.items())

    def get_state(self) -> Any:
        return self._state.copy()

    def set_state(self, state: nx.DiGraph) -> None:
        self._state = state.copy()

    def _reset(self) -> ts.TimeStep:
        self._state = self.init_cluster.copy()
        return ts.restart(
            self.get_state_as_tensors()
        )

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        # Add a new node if needed
        donor, acceptor = action

        # Check that at least one of the atoms is not in the graph
        if sum(i in self._state.nodes for i in action) == 0:
            logger.warning('At least one atom must be in the graph')
            return ts.termination(
                self.get_state_as_tensors(),
                reward=0
            )

        if max(donor, acceptor) >= len(self._state):
            # Add a new node
            new_id = len(self._state)
            self._state.add_node(new_id, label='O')

            # Record the id of the new water. Ensures
            if donor > acceptor:
                donor = new_id
            else:
                acceptor = new_id

        # Update the water cluster
        self._state.add_edge(donor, acceptor, label='donate')
        self._state.add_edge(acceptor, donor, label='accept')

        # Check if the graph is valid
        if not graph_is_valid(self._state, coarse=True):
            logger.warning('Action created an invalid graph.')
            return ts.termination(
                self.get_state_as_tensors(),
                reward=0
            )

        # Compute the reward and if we are done
        reward = self.reward_fn(self._state)
        done = len(self._state) > self.maximum_size

        # Compute the fingerprints for the state
        if done:
            return ts.termination(
                self.get_state_as_tensors(),
                reward=reward
            )
        else:
            return ts.transition(
                self.get_state_as_tensors(),
                reward=reward,
                discount=self.discount_factor
            )
