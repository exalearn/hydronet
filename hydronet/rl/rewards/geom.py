"""Rewards based on graph geometry"""

import numpy as np
import networkx as nx
#import tensorflow as tf
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
    '''
    def __getstate__(self):
        state = self.__dict__.copy()

        # Convert the model to a JSON description and weights
        state['model_weights'] = self.model.get_weights()
        state['model'] = self.model.to_json()

        return state

    def __setstate__(self, state):
        state = state.copy()

        # Convert the MPNN model back to a Keras object
        state['model'] = tf.keras.models.model_from_json(state['model'], custom_objects=custom_objects)
        state['model'].set_weights(state.pop('model_weights'))

        self.__dict__.update(state)
    '''
    def _call(self, graph: nx.Graph) -> float:
        graph = graph.to_undirected()
        n_cycles = self.n_tetramers*len(find_rings(graph, 4)) + self.n_pentamers*len(find_rings(graph, 5)) + self.n_hexamers*len(find_rings(graph, 6))
        return -1*n_cycles 

"""
1) get distribution from database for cluster_size
2) for each cycle size: n_cycles * database_distribution

"""
