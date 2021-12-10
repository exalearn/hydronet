from hydronet.importing import create_graph, make_entry, make_tfrecord, make_nfp_network
from functools import partial
from multiprocessing import Pool
from random import Random
from tqdm import tqdm
from io import StringIO
import tensorflow as tf
import pandas as pd
import numpy as np
import zipfile
import json
import gzip
import os
import networkx as nx
from typing import Dict
import argprase

parser = argparse.ArgumentParser()
parser.add_argument('--save-dir', required=True, help='Directory in which to save output', type=str)
parser.add_argument('--data', required=True, help='Path to csv containing graphs', type=str)
parser.add_argument('--batch-size', default=8192, help='Batch size for processing data', type=int)
args = parser.parse_args()


def make_nfp_network_from_df(idx, df=df, coarsen=True):
    """Make an NFP-compatible network description from an ASE atoms
    
    Args:
        df (pandas df): Dataframe containing graph object
        idx (int): Index in DF
    Returns:
        (dict) Water cluster in NFP-ready format
    """

    # Make the graph from the atoms
    g = df.iloc[idx].graph

    # Write it to a dictionary
    entry = create_inputs_from_nx(g)
    
    entry['energy'] = df.iloc[idx].energy
    return entry



def geometry_record_df(df, idx, coarse=True):
    """Create a JSON-ready record with the geometry and atomic types
    
    Args:
        atoms: ASE atoms object
    Returns:
        dictionary that can be easily serialized to JSON
    """
    g = df.iloc[idx].graph
    
    if coarse:
        z=[8]*len(g.nodes)

    return {
        'z': z,
        'n_water': len(z) // 3,
        'n_atoms': len(z),
        'atom': list(map([8, 1].index, z)),
        'coords': np.stack([i for k,i in nx.get_node_attributes(g,'coords').items()]).tolist(),
        'energy': df.iloc[idx].energy
    }

def create_inputs_from_nx(g: nx.Graph, 
                          atom_types=dict((l, i) for i, l in enumerate(['O', 'H'])),
                          bond_types=dict((l, i) for i, l in enumerate(['donate', 'accept']))):
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


rng_split = np.random.RandomState(4)
rng_pull = np.random.RandomState(129)

# load random graph pickle
df = pd.read_pickle(args.data)
# remove disconnected graphs
df=df.loc[df.shortest_path>0].copy()


# Create the output files
filenames = [
    'geom_train', 'geom_test', 'geom_valid',
    'coarse_train', 'coarse_test', 'coarse_valid'
]

make_output = lambda x: tf.io.TFRecordWriter(os.path.join(args.savedir, f'{x}.proto'))
output_files = dict((x, make_output(x)) for x in filenames)
make_output = lambda x: gzip.open(os.path.join(args.savedir, f'{x}.json.gz'), 'wt')
json_outputs = dict((x, make_output(x)) for x in filenames)

# Control functions
total_entries=len(df)
counter = tqdm(total=total_entries)
coarse_fun = partial(make_nfp_network_from_df, coarsen=True)

try:
    done = False
    c=0
    with Pool(n_jobs - 1) as p:  # One CPU open for serialization
        while not done:
            
            if c >= total_entries:
                done = True
                break
                
            # Get the next batch of entries 
            if c+args.batch_size <= total_entries:
                end = c+args.batch_size
            else:
                end = total_entries
            
            batch = list(range(c,end))
            c=end

            
            # Make the random choices
            split_rnd = rng_split.random(len(batch))
            
            # Save the geometries
            name = 'geom'
            for idx, r in zip(batch, split_rnd):
                # Make the record
                entry = geometry_record_df(df, idx)
                serial_entry = make_tfrecord(entry)
                
                # Store in a specific dataset
                if r < val_fraction:
                    out_name = f'{name}_valid'
                elif r < val_fraction + test_fraction:
                    out_name = f'{name}_test'
                else:
                    out_name = f'{name}_train'
                    
                # Save to file
                output_files[out_name].write(serial_entry)
                print(json.dumps(entry), file=json_outputs[out_name])    
                
            
            # Process for coarse network
            for name, func in zip(['coarse'], [coarse_fun]):
                for idx, entry, r in zip(batch, p.imap(func, batch, chunksize=64), split_rnd):
                    # Serialize the entry
                    serial_entry = make_tfrecord(entry)

                    # Store in a specific dataset
                    if r < val_fraction:
                        out_name = f'{name}_valid'
                    elif r < val_fraction + test_fraction:
                        out_name = f'{name}_test'
                    else:
                        out_name = f'{name}_train'
                    
                    # Save to file
                    output_files[out_name].write(serial_entry)
                    print(json.dumps(entry), file=json_outputs[out_name])
                        
            # Update TQDM
            counter.update(len(batch))
finally:
    for out in output_files.values():
        out.close()
    for out in json_outputs.values():
        out.close()

