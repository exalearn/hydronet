"""Utilities for working with HydroNet data"""

from typing import Union

import networkx as nx


def graph_from_dict(record: dict) -> Union[nx.Graph, nx.DiGraph]:
    """Generate a networkx object from a dictionary describing a graph
    
    Args:
        record (dict): Record to restore from a graph
    Returns:
        A nx.Graph with all atoms for the "atomic" graph or
        a nx.DiGraph with only waters for the "coarse" graph
    """
    
    # Detect if it is a coarse graph
    is_coarse = record['n_water'] == record['n_atom']
    
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
