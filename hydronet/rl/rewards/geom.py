"""Rewards based on graph geometry"""

import numpy as np
import networkx as nx
from hydronet.geometry import find_rings
from hydronet.rl.rewards import RewardFunction

class CyclesReward(RewardFunction):
    """Reward function that assigns reward based on cycle count"""

    def __init__(self, weight: bool = False):
        """
        Args:
            weight: Whether to weight cycles against database distribution
        """
        super().__init__(weight)
        if weight:
            self.n_tetramers = from_dict #TODO
            self.n_pentamers = from_dict #TODO
            self.n_hexamers = from_dict #TODO
        else:
            self.n_tetramers, self.n_pentamers, self.n_hexamers = 1, 1, 1
    def _call(self, graph: nx.Graph) -> float:
        graph = graph.to_undirected()
        n_cycles = self.n_tetramers*len(find_rings(graph, 4)) + self.n_pentamers*len(find_rings(graph, 5)) + self.n_hexamers*len(find_rings(graph, 6))
        return -1*n_cycles 


class ASPLReward(RewardFunction):
    """Reward function that assigns reward based on average shortest path length"""

    def __init__(self):
        super().__init__()
    def _call(self, graph: nx.Graph) -> float:
        '''Higher value = more linear'''
        aspl = nx.average_shortest_path_length(graph.to_undirected())
        return aspl
