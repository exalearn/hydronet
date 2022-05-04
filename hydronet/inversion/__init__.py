"""Functions for inverting a structure from graph to coordinates"""
import os
from typing import Callable
from time import perf_counter

import ase
import numpy as np
import networkx as nx
from ase.optimize import BFGS
from scipy.spatial.transform import Rotation

from hydronet.importing import create_graph, coarsen_graph
from ttm.ase import TTMCalculator

_ttm = TTMCalculator()


def evaluate_inversion(function: Callable[[nx.DiGraph], ase.Atoms], starting: ase.Atoms, relax_ttm: bool = True) -> dict:
    """Evaluate the effectiveness of an inversion technique

    First generates a directed graph from a 3D geometry, then calls an inversion function.
    Returns the comparison of the generated structure to the original

    Args:
        function: Function that takes a directed graph and guesses a 3D geometry
        starting: Starting structure
        relax_ttm: Whether to relax the structure with TTM afterwards
    Returns:
        Performance information:
            - `is_isometric`: Whether the graphs match
            - `adj_difference`: Sum of differences in adjacency matrix of water clusters
            - `unrelaxed_energy`: Energy before relaxation with TTM, if relaxation is performed
            - `new_energy`: Energy after relaxation
            - `energy_diff`: The difference in the energy between the initial and reconstructed clusters
            - `rmsd`: Root mean squared deviation between the initial and reconstructed geometries
            - `invert_time`: How long it too to invert
            - `relax_time`: Time to perform the relaxation
            - `total_time`: How long it took to run the inversion and relaxation

    """

    # Make a directed graph from starting geometry
    starting_graph = create_graph(starting)
    starting_graph_coarse = coarsen_graph(starting_graph)
    starting_energy = _ttm.get_potential_energy(starting)

    # Run the inversion function
    output = {}
    start_time = perf_counter()
    new_geom = function(starting_graph_coarse)
    run_time = perf_counter() - start_time
    output['invert_time'] = run_time

    # If requested, relax the structure
    start_time = perf_counter()
    if relax_ttm:
        # Measure the energy pre-relaxation
        output['unrelaxed_energy'] = _ttm.get_potential_energy(new_geom)

        # Run the relaxation
        new_geom.set_calculator(_ttm)
        dyn = BFGS(new_geom, logfile=os.devnull)
        dyn.run(fmax=0.05, steps=1024)
        new_geom.set_calculator()  # Clear it
    run_time = perf_counter() - start_time
    output['relax_time'] = run_time
    output['total_time'] = output['relax_time'] + output['invert_time']

    # Get the energy of the structure
    new_energy = _ttm.get_potential_energy(new_geom)
    output['final_energy'] = new_energy
    output['energy_diff'] = new_energy - starting_energy

    # Get the graph from the new structure
    new_graph = create_graph(new_geom)
    output['is_isometric'] = nx.is_isomorphic(new_graph, starting_graph)

    new_graph_coarse = coarsen_graph(new_graph)
    output['adj_difference'] = measure_adj_difference(new_graph_coarse, starting_graph_coarse)

    # Compute the RMSD
    _, rmsd = Rotation.align_vectors(starting.positions, new_geom.positions)[:2]
    output['rmsd'] = rmsd

    return output


def measure_adj_difference(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> int:
    """Measure the number of different bonds between two graphs of water clusters

    Args:
        graph_a, graph_b: Two graphs to compare in "coarse" format
    Returns:
        Number of different
    """
    new_adj = nx.to_numpy_array(graph_a.to_undirected(), nodelist=sorted(graph_a.nodes()), weight='label_id', dtype=int)
    old_adj = nx.to_numpy_array(graph_b.to_undirected(), nodelist=sorted(graph_b.nodes()), weight='label_id', dtype=int)
    return int((new_adj != old_adj).sum()) // 2
