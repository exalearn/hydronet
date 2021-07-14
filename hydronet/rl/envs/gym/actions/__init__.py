"""Different choices for actions to use in molecular design"""
from typing import Optional, Sequence, List, Tuple

import numpy as np
import networkx as nx
from gym.spaces import Space


class WaterClusterActions(Space):
    """Action space optimize water cluster networks. Operates on a "coarse graph" that contains only the oxygen atoms

    The only permissible actions are to add a bond between two existing waters or an existing water and a new one.
    Actions are described as the index of the donor and acceptor atom that define the new bond.
    Existing waters may each donate or receive two or fewer hydrogen bonds.
    We provide a

    The space of possible actions changes with the state of the system. Call :meth:`update_actions` to adjust the
    action space based on the current graph.
    """

    def __init__(self):
        super().__init__((2,), np.int)
        self.n = None
        self.current_state: Optional[nx.DiGraph] = None

    def contains(self, x: Tuple[int, int]):
        is_donor = self.get_donor_mask()[x[0]]
        is_acceptor = self.get_acceptor_mask()[x[1]]
        return is_donor and is_acceptor and x[0] != x[1] and not self.current_state.has_edge(x[0], x[1])

    def sample(self):
        # Choose a donor
        donor_options = [i for i, x in enumerate(self.get_donor_mask()) if x]
        donor = self.np_random.choice(donor_options)

        # Choose an acceptor
        acceptor_choices = set(i for i, x in enumerate(self.get_acceptor_mask()) if x and i != donor)
        if donor < len(self.current_state) and self.current_state.degree[donor] > 0:
            acceptor_choices.difference_update(list(self.current_state.neighbors(donor)))
        acceptor = self.np_random.choice(list(acceptor_choices))

        return np.array([donor, acceptor])

    def get_donor_mask(self) -> List[bool]:
        """Get whether each atom in the graph are able to donate bonds

        Note that we include N+1 outputs because the "new atom" can always donate.

        Returns:
            Whether each atom is able to donate a hydrogen bond
        """
        output = [True] * (self.n + 1)
        for node in self.current_state:
            bonds = len(list(i for i, j, d in self.current_state.edges(node, data=True) if d['label'] == 'donate'))
            if bonds >= 2:
                output[node] = False
        return output + [True]

    def get_acceptor_mask(self) -> List[bool]:
        """Get whether each atom in the graph are able to accept bonds

        Note that we include N+1 outputs because the "new atom" can always accept.

        Returns:
            Whether each atom is able to accept a hydrogen bond
        """
        output = [True] * (self.n + 1)
        for node in self.current_state:
            bonds = len(list(i for i, j, d in self.current_state.edges(node, data=True) if d['label'] == 'accept'))
            if bonds >= 2:
                output[node] = False
        return output

    def update_actions(self, new_state: nx.Graph):
        """Update the action space given the current state

        Args:
            new_state (str): Molecule used to define action space
        """
        self.current_state = new_state
        self.n = len(new_state) + 1
