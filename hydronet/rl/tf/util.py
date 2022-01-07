"""Operations useful for running and understanding the RL agent"""
import json
from pathlib import Path
from typing import Callable, Union, List

import networkx as nx
import numpy as np
from scipy.special import logsumexp
from tf_agents.trajectories import Trajectory, StepType
from tf_agents.typing.types import NestedTensor

from hydronet.importing import create_inputs_from_nx


def graphs_from_observation_tensor(obs: NestedTensor) -> List[nx.DiGraph]:
    """Generate a networkx object from a dictionary describing a graph

    Args:
        obs (dict): Observe
    Returns:
        A list of graph representation of the graphs
    """

    # Get the information we need to rebuild the graph
    n_atoms = obs['n_atoms'].numpy()
    bond_type = obs['bond'].numpy()
    connectivity = obs['connectivity'].numpy()

    # Rebuild
    return _graphs_from_obs_data(n_atoms, bond_type, connectivity)


def _graphs_from_obs_data(n_atoms: np.ndarray, bond_type: np.ndarray, connectivity: np.ndarray) -> List[nx.DiGraph]:
    """Build a graph from the state variables in the n_atoms array

    Args:
        n_atoms: Number of atoms in batch
        bond_type: Type of each bond
        connectivity: Type of each bond
    Returns:
        List of directed graphs
    """

    # If we have only one graph in the batch, expand the size of the arrays
    if n_atoms.ndim == 0:
        n_atoms = np.expand_dims(n_atoms, axis=0)
        bond_type = np.expand_dims(bond_type, axis=0)
        connectivity = np.expand_dims(connectivity, axis=0)

    output = []
    # Add the nodes
    for na, bt, cn in zip(n_atoms, bond_type, connectivity):
        graph = nx.DiGraph()
        for i in range(na):
            graph.add_node(i, label='O')

        # Add the bonds
        for t, (a, b) in zip(bt, cn):
            if a >= na or b > na: # This is a placeholder bond
                continue
            label = 'donate' if t == 0 else 'accept'
            graph.add_edge(a, b, label=label)
        output.append(graph)

    return output


class DriverLogger(Callable):
    """Tool for logging the steps taken during a reinforcement learning run

    Saves the chosen action, current list of bonds
    """

    def __init__(self, output_file: Union[str, Path]):
        """
        Parameters
        ----------
        output_file:
            Path to the output file. Should have a .json extension
        """
        self.output_file = Path(output_file)
        self.output_fp = self.output_file.open('w')

    def __call__(self, batch: Trajectory):
        # Get the actions and the reward
        actions = batch.action.numpy().tolist()
        rewards = batch.reward.numpy().tolist()
        step_types = batch.step_type.numpy().tolist()

        # Get the current states as a series of graphs
        graphs = graphs_from_observation_tensor(batch.observation)

        # Get the probability of different moves
        logits = batch.policy_info['dist_params']['logits'].numpy()
        logit_sum = logsumexp(logits, axis=(-1, -2), keepdims=True)
        probs = np.exp(logits - logit_sum).tolist()

        # Write them out to disk
        for action, graph, prob, reward, step_type in zip(actions, graphs, probs, rewards, step_types):
            output = create_inputs_from_nx(graph)
            output['action'] = action
            output['move_probs'] = prob
            output['step_type'] = step_type
            output['reward'] = reward
            print(json.dumps(output), file=self.output_fp)
