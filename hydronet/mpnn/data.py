"""Utilities for creating a data-loader"""

import tensorflow as tf
import numpy as np


if tf.__version__ < '1.15.0':
    from tensorflow.python.ops.ragged.ragged_util import repeat
else:
    repeat = tf.repeat


def parse_records(example_proto):
    """Parse data from the TFRecord"""
    features = {
        'energy': tf.io.FixedLenFeature([], tf.float32, default_value=np.nan),
        'n_atoms': tf.io.FixedLenFeature([], tf.int64),
        'n_bonds': tf.io.FixedLenFeature([], tf.int64),
        'connectivity': tf.io.VarLenFeature(tf.int64),
        'atom': tf.io.VarLenFeature(tf.int64),
        'bond': tf.io.VarLenFeature(tf.int64),
    }
    batch = tf.io.parse_example(example_proto, features)

    # Reshape the bond, connectivity, and node lists
    for c in ['atom', 'bond', 'connectivity']:
        batch[c] = batch[c].flat_values
    return batch


def prepare_for_batching(dataset):
    """Make the variable length arrays into RaggedArrays.
    
    Allows them to be merged together in batches"""
    for c in ['atom', 'bond', 'connectivity']:
        expanded = tf.expand_dims(dataset[c].values, axis=0, name=f'expand_{c}')
        dataset[c] = tf.RaggedTensor.from_tensor(expanded)
    return dataset


def combine_graphs(batch):
    """Combine multiple graphs into a single network"""

    # Compute the mappings from bond index to graph index
    batch_size = tf.size(batch['n_atoms'], name='batch_size')
    mol_id = tf.range(batch_size, name='mol_inds')
    batch['node_graph_indices'] = repeat(mol_id, batch['n_atoms'], axis=0)
    batch['bond_graph_indices'] = repeat(mol_id, batch['n_bonds'], axis=0)

    # Reshape the connectivity matrix to (None, 2)
    batch['connectivity'] = tf.reshape(batch['connectivity'], (-1, 2))

    # Denote the shapes for the atom and bond matrices
    #  Only an issue for 1.14, which cannot infer them it seems
    for c in ['atom', 'bond']:
        batch[c].set_shape((None,))

    # Compute offsets for the connectivity matrix
    offset_values = tf.cumsum(batch['n_atoms'], exclusive=True)
    offsets = repeat(offset_values, batch['n_bonds'], name='offsets', axis=0)
    batch['connectivity'] += tf.expand_dims(offsets, 1)

    return batch


def make_training_tuple(batch, target_name='energy'):
    """Get the output tuple.
    
    Makes a tuple dataset with the inputs as the first element
    and the output energy as the second element
    """

    inputs = {}
    output = None
    for k, v in batch.items():
        if k != target_name:
            inputs[k] = v
        else:
            output = tf.expand_dims(v, 1)
    return inputs, output


def make_data_loader(file_path, batch_size=32, shuffle_buffer=None, 
                     n_threads=tf.data.experimental.AUTOTUNE, shard=None,
                     cache: bool = False) -> tf.data.TFRecordDataset:
    """Make a data loader for tensorflow
    
    Args:
        file_path (str): Path to the training set
        batch_size (int): Number of graphs per training batch
        shuffle_buffer (int): Width of window to use when shuffling training entries
        n_threads (int): Number of threads over which to parallelize data loading
        cache (bool): Whether to load the whole dataset into memory
        shard ((int, int)): Parameters used to shared the dataset: (size, rank)
    Returns:
        (tf.data.TFRecordDataset) An infinite dataset generator
    """

    r = tf.data.TFRecordDataset(file_path)

    # Save the data in memory if needed
    if cache:
        r = r.cache()
        
    # Shuffle the entries
    if shuffle_buffer is not None:
        r = r.shuffle(shuffle_buffer)
        
    # Shard after shuffling (so that each rank will be able to make unique batches each time)
    if shard is not None:
        r = r.shard(*shard)

    # Add in the data preprocessing steps
    #  Note that the `batch` is the first operation
    r = r.batch(batch_size).map(parse_records, n_threads).map(prepare_for_batching, n_threads)

    # Return full batches
    return r.map(combine_graphs, n_threads).map(make_training_tuple)
