"""Choices for search spaces"""

import networkx as nx
from gym import Space

from hydronet.data import graph_is_valid


class AllValidClusters(Space):
    """An observation space that contains all valid water clusters"""

    def sample(self):
        raise NotImplementedError('This design space does not support sampling')

    def contains(self, x: nx.DiGraph):
        # Make sure basic consistency things are met
        if not graph_is_valid(x, coarse=True):
            return False

        # Molecules must donate or receive nor more than 2 bonds
        for node in x.nodes:
            n_donate = 0
            n_accept = 0
            for b in x[node]:
                if x.get_edge_data(node, b)['label'] == 'donate':
                    n_donate += 1
                else:
                    n_accept += 1

            if n_donate > 2 or n_accept > 2:
                return False

        return True
