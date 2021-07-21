import tensorflow as tf
from keras.layers import Embedding, Softmax, Dense
from tensorflow_probability.python.distributions import Categorical
from tf_agents.trajectories import time_step
from tf_agents.typing.types import NestedTensor, TimeStep
from tf_agents.specs import tensor_spec

from hydronet.mpnn.data import combine_graphs
from hydronet.mpnn.layers import MessageBlock

from tf_agents.networks import network


class GCPN(network.Network):
    """Graph convolutional policy network.

    Implemented using a simple message-passing framework for now"""

    def __init__(self, observation_spec, action_spec, example_timestep: TimeStep,
                 num_messages: int = 1, node_features: int = 32):
        """

        Args:
            observation_spec: Specification for the observed space
            action_spec: Specification for the observed output space
            example_timestep: Example timestep used when initializing the network
            num_messages: Number of message-passing steps
            node_features: Number of features to use to represent a node
        """
        super().__init__(
            input_tensor_spec=observation_spec,
            state_spec=(),
            name='GCPN'
        )

        # Store the example_timestep
        self.example_timestep = example_timestep.observation
        self.action_spec = action_spec

        # Make sure the input spec has the required fields
        for f in ['n_atoms', 'atom', 'bond', 'connectivity', 'allowed_actions']:
            assert f in observation_spec, f'Observation spec is missing {f}'
        assert action_spec.shape == (2,), 'Action spec is the wrong shape. Should be (2,)'

        # Make the layers needed by the computation
        self.node_features = node_features
        self.bond_embedding = Embedding(2, node_features)
        self.message_layers = [MessageBlock(atom_dimension=node_features, name=f'message_{i}')
                               for i in range(num_messages)]
        self.softmax = Softmax(axis=1)
        self.donor_dense = [Dense(node_features // 2, name=f'donor_{i}') for i in range(2)]
        self.donor_dense.append(Dense(1, name='donor_last'))
        self.acceptor_dense = [Dense(node_features // 2, name=f'acceptor_{i}') for i in range(2)]
        self.acceptor_dense.append(Dense(1, name='acceptor_last'))

    def create_variables(self, input_tensor_spec=None, **kwargs):
        # TF-Agents generates a random input to run through the network,
        #  which does not work with our MPNN. The message passing layers
        #  do not work with every layer that is
        initial_state = self.get_initial_state(batch_size=1)
        step_type = tf.fill((1,), time_step.StepType.FIRST)
        outputs = self.__call__(
            self.example_timestep,
            step_type=step_type,
            network_state=initial_state,
            **kwargs)

        self._network_output_spec = tensor_spec.remove_outer_dims_nest(
            tf.type_spec_from_value(outputs[0]), num_outer_dims=1)
        return self._network_output_spec

    def convert_env_to_mpnn_batch(self, batch: [str, tf.Tensor]) -> NestedTensor:
        """Convert a batch from the environment into the form needed for our message-passing network

        Args:
            batch: Batch from a dataset of observations
        Returns:
            Data in a format that is usable by the MPNN layers
        """

        # Make a copy of the batch
        batch = batch.copy()

        # Set the number of atoms and bonds to include the dummy nodes/edges
        batch_size, max_atoms = tf.shape(batch['atom'])
        _, max_bonds = tf.shape(batch['bond'])
        batch['n_atoms'] = tf.zeros((batch_size,), dtype=tf.int32) + max_atoms
        batch['n_bonds'] = tf.zeros((batch_size,), dtype=tf.int32) + max_bonds

        # Flatten the atom and bond arrays
        batch['atom'] = tf.reshape(batch['atom'], (batch_size * max_atoms))
        batch['bond'] = tf.reshape(batch['bond'], (batch_size * max_bonds))

        return combine_graphs(batch)

    def call(self, observations, step_type=(), network_state=()):
        """Perform a few graph message passing steps"""
        allowed_actions = observations['allowed_actions']

        # Get the number of nodes per graph
        batch_size, nodes_per_graph = tf.shape(observations['atom'])

        # Prepare the data in a form ready for the data
        batch = self.convert_env_to_mpnn_batch(observations)

        # Make initial features for the atoms and bonds
        bond_features = self.bond_embedding(batch['bond'])
        atom_features = tf.ones((tf.shape(batch['atom'])[0], self.node_features), dtype=tf.float32)

        # Perform the message steps
        for message_layer in self.message_layers:
            atom_features, bond_features = message_layer([atom_features, bond_features, batch['connectivity']])

        # Reshape the atom features so they are arranged (cluster, atom, feature)
        atom_features = tf.reshape(atom_features, (batch_size, nodes_per_graph, self.node_features))

        # Determine the source node: Use the atom_features
        can_be_donor = tf.reduce_any(allowed_actions > 0, axis=1)  # If *any* bonds are possible with this as a donation
        donor_features = atom_features
        for dense in self.donor_dense:
            donor_features = dense(donor_features)
        donor_prob = self.softmax(donor_features[:, :, 0], can_be_donor)  # Assign a prob=0 for invalid donors
        donor_choice = Categorical(probs=donor_prob).sample()

        # Determine the destination nodes: Use features of donor atom and the possible acceptors
        donor_features = tf.gather(atom_features, donor_choice, axis=1, batch_dims=1)  # Get features for donor
        valid_acceptors = tf.gather(allowed_actions, donor_choice, axis=1, batch_dims=1)  # Get possible acceptors
        acceptor_features = tf.concat([
            tf.tile(donor_features[:, None, :], (1, nodes_per_graph, 1)),
            atom_features
        ], axis=2, name='accept_features')  # Acceptor features are those of the donor and each possible acceptor
        for dense in self.acceptor_dense:
            acceptor_features = dense(acceptor_features)
        acceptor_prob = self.softmax(acceptor_features[:, :, 0], valid_acceptors)  # Assign a prob=0 invalid acceptors
        acceptor_choice = Categorical(probs=acceptor_prob).sample()

        return tf.stack([donor_choice, acceptor_choice], axis=1, name='output_stack'), network_state
