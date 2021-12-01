"""Functions for inverting a structure from graph to coordinates"""
import os
from typing import Callable
from time import perf_counter

import ase
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
            - `energy_diff`: The difference in the energy between the initial and reconstructed clusters
            - `rmsd`: Root mean squared deviation between the initial and reconstructed geometries
            - `run_time`: How long it took to run the inversion
    """

    # Make a directed graph from starting geometry
    starting_graph = create_graph(starting)
    starting_graph_coarse = coarsen_graph(starting_graph)
    starting_energy = _ttm.get_potential_energy(starting)

    # Run the inversion function
    start_time = perf_counter()
    new_geom = function(starting_graph_coarse)

    # If requested, relax the structure
    if relax_ttm:
        # Run the relaxation
        new_geom.set_calculator(_ttm)
        dyn = BFGS(new_geom, logfile=os.devnull)
        dyn.run(fmax=0.05)
    run_time = perf_counter() - start_time

    # Get the energy of the structure
    new_energy = _ttm.get_potential_energy(new_geom)

    # Get the graph from the new structure
    new_graph = create_graph(new_geom)

    # Compute the RMSD
    _, rmsd = Rotation.align_vectors(starting.positions, new_geom.positions)

    return {
        'is_isometric': nx.is_isomorphic(new_graph, starting_graph),
        'energy_diff': new_energy - starting_energy,
        'rmsd': rmsd,
        'run_time': run_time
    }
