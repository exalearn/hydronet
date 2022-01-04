import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import networkx as nx
import json
from utils import randomg
from multiprocessing import Pool
from ttm.ase import SciPyFminLBFGSB, TTMCalculator
import os
import os.path as op
import gzip
# install hydronet or add the path to the repo here
sys.path.insert(0, '/people/pope044/Exalearn/hydronet')

# graph gen
from hydronet.importing import create_graph, coarsen_graph
from hydronet.inverting import convert_directed_graph_to_xyz
from hydronet.importing import make_tfrecord

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', required=True, help='Directory in which to save output', type=str)
parser.add_argument('--filename', required=True, help='Filename to save data', type=str)
parser.add_argument('--samples', default=100, help='Number of samples for each cluster size', type=int)
parser.add_argument('--n-jobs', default=12, help='Number of processors to use', type=int)
args = parser.parse_args()

def create_inputs_from_nx(g: nx.Graph, 
                          atom_types=dict((l, i) for i, l in enumerate(['O', 'H'])),
                          bond_types=dict((l, i) for i, l in enumerate(['donate', 'accept']))
                          ):
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
    edge_type_id = np.array(edge_type_id)[inds].tolist()

    return {
        # Determine number of waters based on whether single node or 3 nodes per water
        'n_waters': len(atom_type) if is_digraph else len(atom_type) // 3,
        'n_atoms': len(atom_type),
        'n_bonds': len(edge_type),
        'atom': atom_type_id,
        'bond': edge_type_id,
        'connectivity': connectivity.tolist()
    }


def generate_random_relaxed_tf_entry(n_waters, name = args.filename,
                                     calc=TTMCalculator(), fmax=0.05):

    # generate random graph
    G=randomg.generate_random_graph(n_waters, bidirectional=True)
    
    # convert atoms to coords
    atoms = convert_directed_graph_to_xyz(G)
    
    # add claculator
    atoms.calc = calc

    # relax structure
    dyn = SciPyFminLBFGSB(atoms)
    dyn.run(fmax=fmax)
    
    # check if graph structure changed
    Hr = create_graph(atoms)
    Gr = coarsen_graph(Hr)
    
    if nx.number_connected_components(Gr.to_undirected()) != 1:
        return
    
    else:
        entry = create_inputs_from_nx(Gr)
        entry['energy'] = calc.get_potential_energy(atoms)
        serial_entry = make_tfrecord(entry)
                    
        # Save to file
        output_files[name].write(serial_entry)
        print(json.dumps(entry), file=json_outputs[name])
        
    return 

# Create the output files
filenames = [args.filename]

make_output = lambda x: tf.io.TFRecordWriter(os.path.join(args.save_dir, f'{x}.proto'))
output_files = dict((x, make_output(x)) for x in filenames)
make_output = lambda x: gzip.open(os.path.join(args.save_dir, f'{x}.json.gz'), 'wt')
json_outputs = dict((x, make_output(x)) for x in filenames)

# set up samples
w_samples=np.tile(np.expand_dims(np.arange(10,31),1), args.samples).flatten()
print(f'Total samples to produce: {len(w_samples)}')

with Pool(args.n_jobs) as p: 
    p.map(generate_random_relaxed_tf_entry, w_samples)

for out in output_files.values():
    out.close()
for out in json_outputs.values():
    out.close()    


