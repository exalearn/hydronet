"""Utilities for working with HydroNet data"""

from typing import Union

from ase.calculators.singlepoint import SinglePointCalculator
import networkx as nx
import ase


def graph_is_valid(graph: Union[nx.Graph, nx.DiGraph], coarse: bool) -> bool:
    """Check whether a graph is valid. E.g., it has the expected bond types, number of nodes, etc.

    Args:
        graph: Graph to be checked
        coarse: Whether the graph is coarse or atomic
    Returns:
         (bool) Whether the graph is valid
    """

    if coarse:
        # Make sure that the graph is directional
        if not graph.is_directed():
            return False

        # Make sure the node IDs are between [0, N) (N == number of nodes)
        if set(graph.nodes) != set(range(graph.number_of_nodes())):
            return False

        # Check that all nodes are bonded and that every bond has a matching pair of donor/acceptor
        for node in graph.nodes:
            # Check the number of edges
            edges = graph[node]
            if len(edges) == 0:
                return False

            # Check the bonds
            for b in edges:
                if b == node:
                    return False  # No self-bonding
                out_type = graph.get_edge_data(node, b)['label']
                in_type = graph.get_edge_data(b, node)['label']
                if {out_type, in_type} != {'donate', 'accept'}:
                    return False
    else:
        raise NotImplementedError()

    return True


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
