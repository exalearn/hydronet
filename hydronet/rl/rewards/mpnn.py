"""Rewards based on a MPNN"""

import networkx as nx
import tensorflow as tf
from tensorflow.keras.models import Model

from hydronet.mpnn.inference import run_inference
from hydronet.rl.rewards import RewardFunction
from hydronet.mpnn.layers import custom_objects


class MPNNReward(RewardFunction):
    """Reward function that estimates the energy of a water cluster using an MPNN"""

    def __init__(self, model: Model, maximize: bool = False, big_value: float = 100., per_water: bool = True):
        """
        Args:
            model: Keras MPNN model (trained using the tools in this package)
            maximize: Whether to maximize or minimize the target function
            big_value: Stand-in value to use for compounds the MPNN fails on
            per_water: Whether to return the energy per water
        """
        super().__init__(maximize)
        self.model = model
        self.per_water = per_water
        self.big_value = abs(big_value)
        if self.maximize:
            self.big_value *= -1

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

    def _call(self, graph: nx.Graph) -> float:
        energy = run_inference(self.model, graph)
        return energy / len(graph) if self.per_water else energy
