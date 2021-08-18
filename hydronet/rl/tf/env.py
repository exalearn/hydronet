"""Environments for TF-agents"""
from typing import Dict, Any, List, Tuple
import logging

import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import BoundedArraySpec
from tf_agents.typing import types
import networkx as nx

from hydronet.data import graph_is_valid
from hydronet.importing import create_inputs_from_nx
from hydronet.rl.rewards import RewardFunction, BondCountReward

logger = logging.getLogger(__name__)


class SimpleEnvironment(PyEnvironment):
    """Environment for gradually growing a water cluster. Terminates only when a maximum size has been met

    The state is the state of the graph:

        - ``n_atoms``: Number of atoms. We store this number so that you can determine which entries
            of the connectivity matrix are actually placeholders
        - ``atom_types``: Always 0's, as we currently support only water molecules
        - ``bond_types``: Either 0 (donate) or 1 (accept)
        - ``connectivity``: Defines the connectivity between water clusters (donor, acceptor).
            There are many placeholder entries in here where both the donor and acceptor are
            ``maximum_size + 1``. These are placeholders, which are needed because TF-agents
            need fixed-size arrays, and set to ``maximum_size + 1`` to have tf.gather return
            an array of the correct size.
        - ``allowed_actions``: A matrix where [donor, acceptor] equals 1 if a move is allowed
            and 0 otherwise

    """

    # TODO (wardlt): Harmonize the terms. Make them consistent as graph terms (e.g., source_id, destination_id)

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
        self._episode_ended = False

    def observation_spec(self) -> types.NestedArraySpec:
        # TF-Agents require fixed size arrays. We know the number of bonds cannot be more than 4 times
        #  the number of atoms, as each water can only donate 2 bonds.
        # We use 2 + the maximum number of atoms so that we can include a "ghost" ready to be added to the cluster,
        #  and a "collector" to donate and receive bonds placeholder bonds
        n = self.maximum_size + 2
        return {
            'n_atoms': BoundedArraySpec((), minimum=0, dtype='int32', name='n_atoms'),
            'n_bonds': BoundedArraySpec((), minimum=0, dtype='int32', name='n_bonds'),
            'atom': BoundedArraySpec((n,), minimum=0, maximum=1, dtype='int32'),
            'bond': BoundedArraySpec((n * 4,), minimum=0, maximum=1, dtype='int32'),
            'connectivity': BoundedArraySpec((n * 4, 2), minimum=0, maximum=n, dtype='int32'),
            'allowed_actions': BoundedArraySpec((n, n), minimum=0, maximum=1, dtype='int32')
        }

    def action_spec(self) -> types.NestedArraySpec:
        return BoundedArraySpec((2,), dtype='int32', minimum=0, maximum=self.maximum_size)

    def get_state_as_tensors(self) -> Dict[str, np.ndarray]:
        # Get the data as arrays
        simple_graph = create_inputs_from_nx(self._state)

        # Build the buffered arrays
        n = self.maximum_size + 2
        output = {
            'n_atoms': np.array(simple_graph['n_atoms'], np.int32),
            'n_bonds': np.array(simple_graph['n_bonds'], np.int32),
            'atom': np.zeros((n,), dtype=np.int32),
            'bond': np.zeros((n * 4,), dtype=np.int32),
            'connectivity': np.zeros((n * 4, 2), dtype=np.int32) + self.maximum_size + 1,
            'allowed_actions': self.get_valid_actions_as_matrix()
        }
        for i in ['atom', 'bond']:
            output[i][:len(simple_graph[i])] = simple_graph[i]
        output['connectivity'][:len(simple_graph['connectivity']), :] = np.array(simple_graph['connectivity'])

        return dict((k, np.array(v)) for k, v in output.items())

    def get_state(self) -> Any:
        return self._state.copy()

    def set_state(self, state: nx.DiGraph) -> None:
        self._state = state.copy()

    def _reset(self) -> ts.TimeStep:
        self._state = self.init_cluster.copy()
        self._episode_ended = False
        return ts.restart(
            self.get_state_as_tensors()
        )

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        # If it is a new episode is starting
        if self._episode_ended:
            return self.reset()

        # Add a new node if needed
        donor, acceptor = action

        # Check that at least one of the atoms is not in the graph
        if sum(i in self._state.nodes for i in action) == 0:
            self._episode_ended = True
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
            self._episode_ended = True
            logger.warning('Action created an invalid graph.')
            return ts.termination(
                self.get_state_as_tensors(),
                reward=0
            )

        # Compute the reward and if we are done
        reward = self.reward_fn(self._state)
        done = len(self._state) > self.maximum_size or len(self.get_valid_moves()) == 0

        # Compute the fingerprints for the state
        if done:
            self._episode_ended = True
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

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all possible valid moves

        Returns:
            List of possible actions
        """

        output = []  # Stores the output arrays
        # Get the nodes that have already accepted two bonds
        acc_counts = [0] * len(self._state)
        for u, v, data in self._state.edges(data=True):
            if data['label'] == 'accept':
                acc_counts[u] += 1
        full_acceptors = [i for i, c in enumerate(acc_counts) if c >= 2]

        # Loop over each atom in the graph
        for node in self._state.nodes:
            # Get the points to which this water is already bonded
            edges = self._state[node]

            # Get the waters to which this water donates or accepts bonds
            donating = []
            accepting = []
            for other in edges:
                if self._state.get_edge_data(node, other)['label'] == 'donate':
                    donating.append(other)
                else:
                    accepting.append(other)

            # Populate the lists of possible donations
            if len(donating) < 2:  # Cannot donate more than twice
                max_other = min(len(self._state) + 1, self.maximum_size + 1)
                for other in range(max_other):
                    #  Cannot donate to self, somewhere it has already donated, from where it accepts a bond,
                    #   or from waters that have already accepted two
                    if other != node and other not in donating \
                            and other not in accepting and other not in full_acceptors:
                        output.append((node, other))

        # A new water can donate bonds to all waters that are not yet accepting two bonds
        if len(self._state) <= self.maximum_size:
            new = len(self._state)
            for other in range(new):
                if other not in full_acceptors:
                    output.append((new, other))

        return output

    def get_valid_actions_as_matrix(self) -> np.ndarray:
        """Get the valid actions as a adjacency matrix

        Returns:
            Matrix where value is 0 if move is not possible and 1 if it is
        """

        # Initialize the output
        size = self.maximum_size + 2
        output = np.zeros((size, size), dtype='int32')

        # Store the valid moves
        for d, a in self.get_valid_moves():
            output[d, a] = 1
        return output
