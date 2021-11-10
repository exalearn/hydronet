"""Implementations of actor and critic networks that employ message-passing layers"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense
from tf_agents.specs.distribution_spec import DistributionSpec
from tf_agents.specs import BoundedTensorSpec, TensorSpec
from tf_agents.trajectories import time_step
from tf_agents.typing.types import NestedTensor, TimeStep

from hydronet.mpnn.data import combine_graphs
from hydronet.mpnn.layers import MessageBlock

from tf_agents.networks import network

from hydronet.rl.tf.distribution import MultiCategorical


def convert_env_to_mpnn_batch(batch: [str, tf.Tensor]) -> NestedTensor:
    """Convert a batch from the environment into the form needed for our message-passing network

    Args:
        batch: Batch from a dataset of observations
    Returns:
        Data in a format that is usable by the MPNN layers
    """

    # Make a copy of the batch
    batch = batch.copy()

    # Set the number of atoms and bonds to include the dummy nodes/edges
    batch_info = tf.shape(batch['atom'])
    batch_size = batch_info[0]
    bond_info = tf.shape(batch['bond'])
    batch['n_atoms'] = tf.zeros((batch_size,), dtype=tf.int32) + batch_info[1]
    batch['n_bonds'] = tf.zeros((batch_size,), dtype=tf.int32) + bond_info[1]

    # Flatten the atom and bond arrays
    batch['atom'] = tf.reshape(batch['atom'], (batch_size * batch_info[1],))
    batch['bond'] = tf.reshape(batch['bond'], (batch_size * bond_info[1],))

    return combine_graphs(batch)


def _unstack_observations(observations):
    """Flatten observations so that there is a single batch dimension

    Returns
    -------
    batch_size: int
        Number of observations in this batch
    observations: NestedTensor
        Re-shaped version of the observations
    outer_shape:
        The shape of the batch dimensions
    """
    outer_shape = observations['allowed_actions'].shape[:-3]
    batch_size = observations['allowed_actions'].shape[-3]
    new_batch_size = tf.reduce_prod(outer_shape) * batch_size

    def _flatten(x):
        old_shape = tf.shape(x)
        new_shape = tf.concat(
            ([new_batch_size], old_shape[outer_shape.ndims + 1:]), axis=0
        )
        return tf.reshape(x, new_shape)

    observations = tf.nest.map_structure(_flatten, observations)
    return batch_size, observations, outer_shape


class MPNNNetworkMixin:
    """Provides the attributes and functions to perform message passing needed for a MPNN-based network"""

    def __init__(self, *args, node_features: int = 32, num_messages: int = 1, output_layers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_features = node_features
        self.output_dense = [Dense(node_features * 2, activation='tanh', name=f'output_{i}')
                             for i in range(output_layers)]
        self.output_dense.append(Dense(1, name='output_last'))
        self.bond_embedding = Embedding(2, node_features)
        self.message_layers = [MessageBlock(atom_dimension=node_features, name=f'message_{i}')
                               for i in range(num_messages)]

    @tf.function
    def perform_message_passing(self, observations):
        """Produce features for each node using message passing

        Parameters
        ----------
        observations: Observations for each batch

        Returns
        -------
        atom_features:
            (batch_size, nodes_per_graph, num_features) array of features for each node
        """
        # Get the number of nodes per graph
        batch_info = tf.shape(observations['atom'])

        # Prepare the data in a form ready for the neural network
        batch = convert_env_to_mpnn_batch(observations)

        # Make initial features for the atoms and bonds
        bond_features = self.bond_embedding(batch['bond'])
        atom_features = tf.ones((tf.shape(batch['atom'])[0], self.node_features), dtype=tf.float32)

        # Perform the message steps
        for message_layer in self.message_layers:
            atom_features, bond_features = message_layer([atom_features, bond_features, batch['connectivity']])

        # Reshape the atom features so they are arranged (cluster, atom, feature)
        atom_features = tf.reshape(atom_features, (batch_info[0], batch_info[1], self.node_features))
        return atom_features


class GCPNActorNetwork(MPNNNetworkMixin, network.DistributionNetwork,):
    """Graph convolutional policy network that returns a probability distribution for different actions
    given a certain graph

    Ensures that the only "allowed" actions receive non-zero probabilities. Returns a :class:`MultiCategorical`
    class that describes the probability distribution.
    """

    def __init__(self, observation_spec, action_spec, example_timestep: TimeStep,
                 output_layers: int = 2, num_messages: int = 1,
                 node_features: int = 32, graph_features: bool = True):
        """

        Args:
            observation_spec: Specification for the observed space
            action_spec: Specification for the observed output space
            example_timestep: Example timestep used when initializing the network. DEV NOTE:
                Needed because it is difficult to generate a valid graph from the observation spec
            num_messages: Number of message-passing steps
            node_features: Number of features to use to represent a node
            graph_features: Whether to combine node- and graph-level features (or just node features)
                to describe a pair of nodes. Graph-level features are created by summing over all nodes
        """

        # Store the specifications
        self.example_timestep = example_timestep.observation
        self.action_spec = action_spec
        self.graph_features = graph_features

        # Build the output specification for the network, which define the distribution and samples
        max_nodes = example_timestep.observation['atom'].shape[-1]
        output_spec = DistributionSpec(
            MultiCategorical,
            input_params_spec={
                'logits': BoundedTensorSpec((max_nodes, max_nodes), minimum=0, maximum=1, dtype='float32')
            },
            sample_spec=self.action_spec
        )

        super().__init__(
            output_layers=output_layers,
            num_messages=num_messages,
            node_features=node_features,
            input_tensor_spec=observation_spec,
            state_spec=(),
            output_spec=output_spec,
            name='GCPN'
        )

        # Make sure the input spec has the required fields
        for f in ['n_atoms', 'atom', 'bond', 'connectivity', 'allowed_actions']:
            assert f in observation_spec, f'Observation spec is missing {f}'
        assert action_spec.shape == (2,), 'Action spec is the wrong shape. Should be (2,)'

    def create_variables(self, input_tensor_spec=None, **kwargs):
        # TF-Agents generates a random input to run through the network,
        #  which does not work with our MPNN. The message passing layers
        #  do not work with every layer that is
        initial_state = self.get_initial_state(batch_size=1)
        step_type = tf.fill((1,), time_step.StepType.FIRST)
        self.__call__(
            self.example_timestep,
            step_type=step_type,
            network_state=initial_state,
            **kwargs)

        # We return the action spec to describe the output of the model
        #  DEV NOTE: I'm not sure if this is actually used since we define the output_spec
        #  in the initializer of the DistributionNetwork anyway. *nervous shrug*
        return self.action_spec

    def call(self, observations, step_type=(), network_state=()):
        """Perform a few graph message passing steps"""

        # Reshape the observations to only have one
        batch_size, observations, outer_shape = _unstack_observations(observations)

        # Get the allowed actions
        allowed_actions = observations['allowed_actions']

        # Make features for each atom using message-passing
        atom_features = self.perform_message_passing(observations)

        # Make a Cartesian product of the features source/destination pairs
        nodes_per_graph = tf.shape(observations['atom'])[1]
        pair_features = [
            tf.tile(tf.expand_dims(atom_features, 2), (1, 1, nodes_per_graph, 1)),
            tf.tile(tf.expand_dims(atom_features, 1), (1, nodes_per_graph, 1, 1)),
        ]

        # If desired, add the total state of the network as an input as well
        if self.graph_features:
            valid_atom_features = tf.where(
                tf.expand_dims(observations['is_atom'], axis=-1),
                atom_features,
                0
            )  # Ensures that we do not sum over ghost atoms
            graph_features = tf.reduce_sum(valid_atom_features, axis=1, keepdims=True)
            pair_features.append(
                tf.tile(tf.expand_dims(graph_features, 1), (1, nodes_per_graph, nodes_per_graph, 1))
            )

        # Stack them together to create a single feature for all source/destination pairs
        pair_features = tf.concat(pair_features, axis=3)  # shape: (batch_size, nodes_per_graph, atom_features * [2|3])

        # Pass them through dense layers to get a single value per pair
        pair_values = pair_features
        for layer in self.output_dense:
            pair_values = layer(pair_values)

        # Set the logit for disallowed actions to negative infinity using softmax (making them a probability of 0)
        pair_logits = tf.where(
            allowed_actions == 1,
            pair_values[:, :, :, 0],
            -np.inf
        )

        # Special case: If no actions are allowed, then allow all actions with equal probabilities
        #  Addresses a weird problem: If there are no valid actions then a policy agent will
        #  pick a randomly-selected one, which causes problems when you train the model.
        #  That randomly-selected action will have a probability of 0 and a log-probability of -inf.
        #  Many training algorithms use the log-probability in the loss functions and doing math
        #  with infinities results will result in numerical problems when computing loss.
        no_allowed_actions = tf.reduce_all(allowed_actions == 0, axis=[1, 2], keepdims=True)
        pair_logits = tf.where(
            tf.tile(no_allowed_actions, (1, nodes_per_graph, nodes_per_graph)),
            1.,
            pair_logits
        )

        # Shape the outputs like the original input shape
        #  DEV NOTE: Do the "batch_utils" have support for these kind of operations?
        bond_counts = tf.shape(pair_logits)[-1]
        output_shape = tf.concat((outer_shape, (batch_size, bond_counts, bond_counts)), axis=0)
        pair_logits = tf.reshape(pair_logits, output_shape)
        return MultiCategorical(pair_logits), ()


class GCPNCriticNetwork(MPNNNetworkMixin, network.Network):
    """Produce a value for a certain state using a MPNN

    Uses an "atomic contribution" model where we first compute a value per node
    and express the value of the graph as a sum over all nodes.
    """

    def __init__(self, observation_spec, example_timestep: TimeStep,
                 output_layers: int = 2, num_messages: int = 1,
                 node_features: int = 32):
        """

        Parameters
        ----------
        observation_spec:
            Specification for a timestep
        example_timestep:
            Example time step, used when initializing the network
        output_layers:
            Number of output layers to include in the network
        num_messages:
            Number of message-passing layers
        node_features:
            Number of features to describe each node/edge
        """
        # Initialize the network
        super().__init__(output_layers=output_layers, num_messages=num_messages,
                         node_features=node_features, input_tensor_spec=observation_spec)
        self.example_timestep = example_timestep.observation

    def create_variables(self, input_tensor_spec=None, **kwargs):
        # TF-Agents generates a random input to run through the network,
        #  which does not work with our MPNN. The message passing layers
        #  do not work with every layer that is
        initial_state = self.get_initial_state(batch_size=1)
        step_type = tf.fill((1,), time_step.StepType.FIRST)
        self.__call__(
            self.example_timestep,
            step_type=step_type,
            network_state=initial_state,
            **kwargs)

        return TensorSpec(shape=(1,))

    def call(self, observations, step_type=(), network_state=()):
        """Perform a few graph message passing steps then condense to a single output per graph"""

        # Unpack the observations
        batch_size, observations, outer_shape = _unstack_observations(observations)

        # Run the message passing
        atom_features = self.perform_message_passing(observations)

        # Condense them to a single output per atom
        atom_values = atom_features
        for layer in self.output_dense:
            atom_values = layer(atom_values)

        # Sum over all "real atoms" per graph
        valid_atom_values = tf.where(
            tf.expand_dims(observations['is_atom'], axis=-1),
            atom_values,
            0
        )
        graph_values = tf.reduce_sum(valid_atom_values, axis=1, keepdims=True)

        # Reshape to the original outer_shape
        output_shape = tf.concat((outer_shape, (batch_size,)), axis=0)
        return tf.reshape(graph_values, output_shape), network_state
