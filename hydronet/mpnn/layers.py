"""Layers needed to create the MPNN model.

Taken from the ``tf2`` branch of the ``nfp`` code:

https://github.com/NREL/nfp/blob/tf2/examples/tf2_tests.ipynb
"""
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class MessageBlock(layers.Layer):
    """Message passing layer for MPNNs

    Takes the state of an atom and bond, and updates them by passing messages between nearby neighbors.

    Following the notation of Gilmer et al., the message function sums all of the atom states from
    the neighbors of each atom and then updates the node state by adding them to the previous state.
    """

    def __init__(self, atom_dimension, activation: str = 'tanh', **kwargs):
        """
        Args:
             atom_dimension (str): Number of features to use to describe each atom
             activation (str): Activation function for the message layers
        """
        super(MessageBlock, self).__init__(**kwargs)
        self.activation = activation
        self.atom_bn = layers.BatchNormalization()
        self.bond_bn = layers.BatchNormalization()
        self.bond_update_1 = layers.Dense(2 * atom_dimension, activation='tanh', use_bias=False)
        self.bond_update_2 = layers.Dense(atom_dimension)
        self.atom_update = layers.Dense(atom_dimension, activation='tanh', use_bias=False)
        self.atom_dimension = atom_dimension

    def call(self, inputs):
        original_atom_state, original_bond_state, connectivity = inputs

        # Batch norm on incoming layers
        atom_state = self.atom_bn(original_atom_state)
        bond_state = self.bond_bn(original_bond_state)

        # Gather atoms to bond dimension
        target_atom = tf.gather(atom_state, connectivity[:, 0])
        source_atom = tf.gather(atom_state, connectivity[:, 1])

        # Update bond states with source and target atom info
        new_bond_state = tf.concat([source_atom, target_atom, bond_state], 1)
        new_bond_state = self.bond_update_1(new_bond_state)
        new_bond_state = self.bond_update_2(new_bond_state)

        # Update atom states with neighboring bonds
        source_atom = self.atom_update(source_atom)
        messages = source_atom * new_bond_state
        messages = tf.math.segment_sum(messages, connectivity[:, 0])

        # Add new states to their incoming values (residual connection)
        bond_state = original_bond_state + new_bond_state
        atom_state = original_atom_state + messages

        return atom_state, bond_state

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_dimension': self.atom_dimension,
            'activation': self.activation
        })
        return config


class GraphNetwork(layers.Layer):
    """Layer that implements an entire MPNN neural network

    Creates the message passing layers and also implements reducing the features of all nodes in
    a graph to a single output vector for a molecule.

    The "reduce" portion (also known as readout by Gilmer) can be configured a few different ways.
    One setting is whether the reduction occurs by combining the feature vectors for each atom
    and then using an MLP to determine the molecular properly or to first reduce the atomic feature
    vectors to a scalar with an MLP and then combining the result.
    You can also change how the reduction is performed, such as via a sum, average, or maximum.

    The reduction to a single feature for an entire molecule is produced by summing a single scalar value
    used to represent each atom. We chose this reduction approach under the assumption the energy of a molecule
    can be computed as a sum over atomic energies."""

    def __init__(self, atom_classes: int, bond_classes: int, atom_dimension: int, num_messages: int, 
                 message_layer_activation: str = 'tanh', output_layer_sizes: List[int] = None, 
                 dropout: float = 0.5, **kwargs):
        """
        Args:
             atom_classes (int): Number of possible types of nodes
             bond_classes (int): Number of possible types of edges
             atom_dimension (int): Number of features used to represent a node and bond
             num_messages (int): Number of message passing steps to perform
             output_layer_sizes ([int]): Number of dense layers that map the atom state to energy
             message_layer_activation: Activation function used in message passing layer
             dropout (float): Dropout rate
        """
        super(GraphNetwork, self).__init__(**kwargs)
        self.atom_embedding = layers.Embedding(atom_classes, atom_dimension, name='atom_embedding')
        self.atom_mean = layers.Embedding(atom_classes, 1, name='atom_mean')
        self.bond_embedding = layers.Embedding(bond_classes, atom_dimension, name='bond_embedding')
        self.message_layers = [MessageBlock(atom_dimension, message_layer_activation)
                               for _ in range(num_messages)]
        self.message_layer_activation = message_layer_activation
        
        # Make the output MLP
        if output_layer_sizes is None:
            output_layer_sizes = []
        self.output_layers = [layers.Dense(s, activation='relu', name=f'dense_{i}')
                              for i, s in enumerate(output_layer_sizes)]
        self.output_layer_sizes = output_layer_sizes
        self.output_atomwise_dense = layers.Dense(1)
        
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs):
        atom_types, bond_types, node_graph_indices, connectivity = inputs

        # Initialize the atom and bond embedding vectors
        atom_state = self.atom_embedding(atom_types)
        bond_state = self.bond_embedding(bond_types)

        # Perform the message passing
        for message_layer in self.message_layers:
            atom_state, bond_state = message_layer([atom_state, bond_state, connectivity])

        # Add some dropout before hte last year
        atom_state = self.dropout_layer(atom_state)
        
        # Apply the MLP layers
        for l in self.output_layers:
            atom_state = l(atom_state)

        # Reduce atom to a single prediction
        atom_output = self.output_atomwise_dense(atom_state) + self.atom_mean(atom_types)

        # Sum over all atoms in a mol
        mol_energy = tf.math.segment_sum(atom_output, node_graph_indices)

        return mol_energy

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_classes': self.atom_embedding.input_dim,
            'bond_classes': self.bond_embedding.input_dim,
            'atom_dimension': self.atom_embedding.output_dim,
            'output_layer_sizes': self.output_layer_sizes,
            'message_layer_activation': self.message_layer_activation, 
            'dropout': self.dropout_layer.rate,
            'num_messages': len(self.message_layers)
        })
        return config


class Squeeze(layers.Layer):
    """Wrapper over the tf.squeeze operation"""

    def __init__(self, axis=1, **kwargs):
        """
        Args:
            axis (int): Which axis to squash
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config
    
    
def build_fn(atomic_mean: float, atom_features: int = 32, message_steps: int = 8, **kwargs) -> Model:
    """Construct a full model
    
    Args:
        atomic_mean: Average energy per node (atom)
        atom_features: Number of features per atom/bond
        message_steps: Number of message passing steps
        
    """
    node_graph_indices = layers.Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = layers.Input(shape=(1,), name='atom', dtype='int32')
    bond_types = layers.Input(shape=(1,), name='bond', dtype='int32')
    connectivity = layers.Input(shape=(2,), name='connectivity', dtype='int32')
    
    # Squeeze the node graph and connectivity matrices
    snode_graph_indices = Squeeze(axis=1)(node_graph_indices)
    satom_types = Squeeze(axis=1)(atom_types)
    sbond_types = Squeeze(axis=1)(bond_types)
    
    output = GraphNetwork(2, 2, atom_features, message_steps, **kwargs, name='mpnn')([satom_types, sbond_types, snode_graph_indices, connectivity])
    
    # Make the model
    model = Model(inputs=[node_graph_indices, atom_types, bond_types, connectivity], outputs=output)
    
    # Set the weights for the MPNN layer based on a batch sampled from the training set
    layer = model.get_layer('mpnn')
    weights = layer.atom_mean.get_weights()
    weights[0][:] = atomic_mean
    layer.atom_mean.set_weights(np.array(weights, np.float32))
    
    return model


custom_objects = {
    'GraphNetwork': GraphNetwork,
    'MessageBlock': MessageBlock,
    'Squeeze': Squeeze
}
