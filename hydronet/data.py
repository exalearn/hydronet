"""Utilities for working with HydroNet data"""

from typing import Union

from ase.calculators.singlepoint import SinglePointCalculator
import networkx as nx
import ase


def graph_from_dict(record: dict) -> Union[nx.Graph, nx.DiGraph]:
    """Generate a networkx object from a dictionary describing a graph
    
    Args:
        record (dict): Record from which to restore a graph
    Returns:
        A nx.Graph with all atoms for the "atomic" graph or
        a nx.DiGraph with only waters for the "coarse" graph
    """
    
    # Detect if it is a coarse graph
    is_coarse = record['n_waters'] == record['n_atoms']
    
    # Make the graph
    graph = nx.DiGraph() if is_coarse else nx.Graph()
    
    # Add the nodes
    for i, t in enumerate(record['atom']):
        graph.add_node(i, label='oxygen' if t == 0 else 'hydrogen')
        
    # Add the bonds
    for t, (a, b) in zip(record['bond'], record['connectivity']):
        if is_coarse:
            label = 'donate' if t == 0 else 'accept'
        else:
            label = 'covalent' if t == 0 else 'hydrogen'
        graph.add_edge(a, b, label=label)
        
    return graph


def atoms_from_dict(record: dict) -> ase.Atoms:
    """Generate an ASE Atoms object from the graph dictionary

    Args:
        record: Record from which to generate an Atoms object
    Returns:
        atoms: The Atoms object
    """
    # Make the atoms object
    atoms = ase.Atoms(positions=record['coords'], numbers=record['z'])

    # Add energy, if available
    if 'energy' in record:
        calc = SinglePointCalculator(atoms, energy=record['energy'])
        atoms.set_calculator(calc)

    return atoms
