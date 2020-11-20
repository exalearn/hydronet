"""Utilities related to reading water cluster networks from disk"""

from typing import Dict

from ase import data
import tensorflow as tf
import networkx as nx
import numpy as np


def infer_water_cluster_bonds(atoms):
    """
    Infers the covalent and hydrogen bonds between oxygen and hydrogen atoms in a water cluster.

    Definition of a hydrogen bond obtained from https://aip.scitation.org/doi/10.1063/1.2742385

    Args:
        atoms (ase.Atoms): ASE atoms structure of the water cluster. Atoms list must be ordered
            such that the two covalently bound hydrogens directly follow their oxygen.
    Returns:
        cov_bonds ([(str, str, 'covalent')]): List of all covalent bonds
        h_bonds [(str, str, 'hydrogen')]: List of all hydrogen bonds
    """

    # Make sure the atoms are in the right order
    z = atoms.get_atomic_numbers()
    assert z[:2].tolist() == [8, 1], "Atom list not in (O, H, H) format"
    coords = atoms.positions

    # Get the covalent bonds
    #  Note: Assumes that each O is followed by 2 covalently-bonded H atoms
    cov_bonds = [(i, i + 1, 'covalent') for i in range(0, len(atoms), 3)]
    cov_bonds.extend([(i, i + 2, 'covalent') for i in range(0, len(atoms), 3)])

    # Get the hydrogen bonds
    #  Start by getting the normal to each water molecule
    q_1_2 = []
    for i in range(0, len(atoms), 3):
        h1 = coords[i + 1, :]
        h2 = coords[i + 2, :]
        o = coords[i, :]
        q_1_2.append([h1 - o, h2 - o])
    v_list = [np.cross(q1, q2) for (q1, q2) in q_1_2]

    #  Determine which (O, H) pairs are bonded
    h_bonds = []
    for idx, v in enumerate(v_list):  # Loop over each water molecule
        for index, both_roh in enumerate(q_1_2):  # Loop over each hydrogen
            for h_index, roh in enumerate(both_roh):
                # Get the index of the H and O atoms being bonded
                indexO = 3 * idx
                indexH = 3 * index + h_index + 1

                # Get the coordinates of the two atoms
                h_hbond = coords[indexH, :]
                o_hbond = coords[indexO, :]

                # Compute wehther they are bonded
                dist = np.linalg.norm(h_hbond - o_hbond)
                if (dist > 1) & (dist < 2.8):
                    angle = np.arccos(np.dot(roh, v) / (np.linalg.norm(roh) * np.linalg.norm(v))) * (180.0 / np.pi)
                    if angle > 90.0:
                        angle = 180.0 - angle
                    N = np.exp(-np.linalg.norm(dist) / 0.343) * (7.1 - (0.05 * angle) + (0.00021 * (angle ** 2)))
                    if N >= 0.0085:
                        h_bonds.append((indexO, indexH, 'hydrogen'))

    return cov_bonds, h_bonds


def create_graph(atoms):
    """
    Given a ASE atoms object, this function returns a graph structure with following properties.
        1) Each graph has two graph-level attributes: actual_energy and predicted_energy
        2) Each node represents an atom and has two attributes: label ('O'/'H' for oxygen and hydrogen) and 3-dimensional
           coordinates.
        3) Each edge represents a bond between two atoms and has two attributes: label (covalent or hydrogen) and distance.
    Args:
        atoms (Atoms): ASE atoms object
    Returns:
        (nx.Graph) Networkx representation of the water cluster
    """

    # Compute the bonds
    cov_bonds, h_bonds = infer_water_cluster_bonds(atoms)

    # Add nodes to the graph
    graph = nx.Graph()
    for i, (coord, Z) in enumerate(zip(atoms.positions, atoms.get_atomic_numbers())):
        graph.add_node(i, label=data.chemical_symbols[Z], coords=coord)

    # Add the edges
    edges = cov_bonds + h_bonds
    for a1, a2, btype in edges:
        distance = np.linalg.norm(atoms.positions[a1, :] - atoms.positions[a2, :])
        graph.add_edge(a1, a2, label=btype, weight=distance)
    return graph


def coarsen_graph(in_graph: nx.Graph) -> nx.DiGraph:
    """Create a graph with only one node per water molecule

    Args:
        in_graph: Input graph, which contains both hydrogens and oxygen
    Returns:
         A directed graph with only the oxygen atoms as nodes. Nodes are identified
         as whether that O atoms is part of a water molecule that donates a hydrogen
         bond to another molecule or whether it receives a hydrogen bond
    """

    # Initialize the output graph
    output = nx.DiGraph()

    # Collect information from the previous graph
    bonds = []
    for node, node_data in in_graph.nodes(data=True):
        if node_data['label'] == 'O':
            # If it is an oxygen, make it a node in the new graph
            output.add_node(node // 3, **node_data)  # Make the count from [0, N)
        elif node_data['label'] == 'H':
            # Check if this H participates in H bonding
            donor = acceptor = None  # Stores the donor and acceptor oxygen id
            for neigh in in_graph.neighbors(node):
                neigh_info = in_graph.get_edge_data(node, neigh)
                bond_type = neigh_info['label']
                if bond_type == 'covalent':
                    donor = neigh // 3, neigh_info['weight']
                else:
                    acceptor = neigh // 3, neigh_info['weight']
            if not (donor is None or acceptor is None):
                bonds.append((donor, acceptor))  # Store as donor->acceptor
        else:
            raise ValueError(f'Unrecognized type: {node_data["label"]}')

    # Assign bonds to each water molecule
    for (d, w_d), (a, w_a) in bonds:
        # Add the edges
        output.add_edge(d, a, label='donate', weight=w_d+w_a)
        output.add_edge(a, d, label='accept', weight=w_d+w_a)

    return output


def make_entry(atoms) -> dict:
    """Create a database record for a water cluster
    i ad
    Args:
        atoms (Atoms): ASE Atoms object
    Returns:
        (dict) Record containing:
            'graph': Graph representation of the cluster
            'energy': Energy of the cluster
            'n_waters': Number of water molecules
    """

    return {
        'graph': create_graph(atoms),
        'energy': atoms.get_potential_energy(),
        'n_waters': len(atoms) // 3
    }


def _numpy_to_tf_feature(value):
    """Converts a Numpy array to a Tensoflow Feature
    
    Determines the dtype and ensures the array is at least 1D
    
    Args:
        value (np.array): Value to convert
    Returns:
        (tf.train.Feature): Feature representation of this full value
    """

    # Make sure value is an array, then flatten it to a 1D vector
    value = np.atleast_1d(value).flatten()

    if value.dtype.kind == 'f':
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    elif value.dtype.kind in ['i', 'u']:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        # Just send the bytes (warning: untested!)
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))


_atom_type_lookup = dict((l, i) for i, l in enumerate(['O', 'H']))
_bond_type_lookup = dict((l, i) for i, l in enumerate(['covalent', 'hydrogen']))
_coarse_bond_type_lookup = dict((l, i) for i, l in enumerate(['donate', 'accept']))


def make_nfp_network(atoms, coarsen: bool = False):
    """Make an NFP-compatible network description from an ASE atoms
    
    Args:
        atoms (ase.Atoms): Atoms object of the water cluster
        coarsen: Whether to coarsen graph to 1 node per atom
    Returns:
        (dict) Water cluster in NFP-ready format
    """

    # Make the graph from the atoms
    g = create_graph(atoms)

    # If desired, coarsen
    if coarsen:
        g = coarsen_graph(g)

    # Write it to a dictionary
    entry = create_inputs_from_nx(g, _atom_type_lookup,
                                  _coarse_bond_type_lookup if coarsen else _bond_type_lookup)
    entry['energy'] = atoms.get_potential_energy()
    return entry


def create_inputs_from_nx(g: nx.Graph, atom_types: Dict[str, int],
                          bond_types: Dict[str, int]) -> dict:
    """Create a NFP-compatible input dictionary from from a Networkx object

    Args:
        g: Input as a NetworkX object
        atom_types: Dictionary of atom types to label
        bond_types: Dictionary of bond types to label
    Returns:
        (dict) Input as a network as a networkx object
    """

    # Get the atom types
    atom_type = [n['label'] for _, n in g.nodes(data=True)]
    atom_type_id = list(map(atom_types.__getitem__, atom_type))

    # If undirected, make it into a directed graph
    is_digraph = True
    if not isinstance(g, nx.DiGraph):
        g = g.to_directed()
        is_digraph = False

    # Get the bond types
    connectivity = []
    edge_type = []
    for a, b, d in g.edges(data=True):
        connectivity.append([a, b])
        edge_type.append(d['label'])
    if not is_digraph:
        assert len(edge_type) % 2 == 0  # Must have both directions
    edge_type_id = list(map(bond_types.__getitem__, edge_type))

    # Sort connectivity array by the first column
    #  This is needed for the MPNN code to efficiently group messages for
    #  each node when performing the message passing step
    connectivity = np.array(connectivity)
    inds = np.lexsort((connectivity[:, 1], connectivity[:, 0]))
    connectivity = connectivity[inds, :]

    return {
        # Determine number of waters based on whether single node or 3 nodes per water
        'n_waters': len(atom_type) if is_digraph else len(atom_type) // 3,
        'n_atom': len(atom_type),
        'n_bond': len(edge_type),
        'atom': atom_type_id,
        'bond': edge_type_id,
        'connectivity': connectivity.tolist()
    }


def make_tfrecord(cluster: dict):
    """Make and serialize a TFRecord for in NFP format
    
    Args:
        cluster (dict): Dictionary representation of water cluster network
    Returns:
        (bytes) Water cluster as a sesrialized string
    """

    # Convert the data to TF features
    features = dict((k, _numpy_to_tf_feature(v)) for k, v in cluster.items())

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()
