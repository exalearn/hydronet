import numpy as np
from tf_agents.agents import PPOAgent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import ActorPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs.tensor_spec import add_outer_dims_nest
import tensorflow as tf

from pytest import fixture
from tf_agents.typing.types import NestedTensor

from hydronet.rl.tf.networks import GCPNActorNetwork, convert_env_to_mpnn_batch, GCPNCriticNetwork
from hydronet.rl.tf.agents import ConstrainedRandomPolicy
from hydronet.rl.tf.env import SimpleEnvironment


@fixture()
def env() -> SimpleEnvironment:
    return SimpleEnvironment()


@fixture()
def tf_env(env) -> TFPyEnvironment:
    return TFPyEnvironment(env, check_dims=True)


@fixture()
def example_batch() -> NestedTensor:
    """Example batch used to train an agent"""
    return {
        'action': tf.constant([[1, 2], [2, 4]], dtype=tf.int32),
        'discount': tf.constant([1., 1.], dtype=tf.float32),
        'next_step_type': tf.constant([0, 0], dtype=tf.int32),
        'observation': {
            'allowed_actions': tf.constant([[[0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0],
                                             [0, 0, 0, 1, 1, 0],
                                             [1, 0, 1, 0, 0, 0],
                                             [1, 0, 1, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0]],
                                            [[0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 1, 1, 0],
                                             [0, 0, 0, 0, 0, 0],
                                             [1, 1, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0]]], dtype=tf.int32),
            'atom': tf.constant([[0] * 6] * 2, dtype=tf.int32),
            'bond': tf.constant([[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                dtype=tf.int32),
            'is_atom': tf.constant([[True, True, True, True, True, False]] * 2, dtype=tf.bool),
            'connectivity': tf.constant([[[0, 1], [0, 2], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [3, 1],
                                          [3, 4], [4, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5],
                                          [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
                                         [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [3, 0],
                                          [3, 1], [3, 4], [4, 3], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5],
                                          [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]], dtype=tf.int32),
            'n_atoms': tf.constant([5, 5], dtype=tf.int32),
            'n_bonds': tf.constant([10, 12], dtype=tf.int32)
        },
        'reward': tf.constant([0, 0], dtype=tf.int32),
        'step_type': tf.constant([2, 2], dtype=tf.int32)
    }


def test_random(env, tf_env):
    policy = ConstrainedRandomPolicy(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec()
    )

    # Make an initial step
    init_step = tf_env.reset()
    allowed_adj = env.get_valid_actions_as_matrix()
    action = policy.action(init_step)
    action = np.squeeze(action.action)
    assert allowed_adj[action[0], action[1]] == 1


def test_random_with_driver(tf_env):
    tf_policy = ConstrainedRandomPolicy(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec()
    )

    # Test without buffers
    driver = DynamicStepDriver(tf_env, tf_policy, [], num_steps=128)
    init_ts = tf_env.reset()
    driver.run(init_ts)

    # Test with a buffer
    buffer = TFUniformReplayBuffer(
        tf_policy.trajectory_spec,
        batch_size=tf_env.batch_size
    )
    driver = DynamicStepDriver(tf_env, tf_policy, [buffer.add_batch], num_steps=128)
    driver.run(init_ts)


def test_gcpn_actor_network(tf_env, example_batch):
    network = GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset(), graph_features=True)
    batch = convert_env_to_mpnn_batch(example_batch['observation'])
    network.create_variables()
    assert 'node_graph_indices' in batch
    assert batch['node_graph_indices'].numpy().max() == 1

    action_choice, _ = network.call(example_batch['observation'])
    assert action_choice.sample().shape == (2, 2)

    # See if the network is differentiable
    with tf.GradientTape() as tape:
        dist, _ = network.call(example_batch['observation'])
        max_prob = tf.reduce_max(dist.log_prob(dist.mode()))
    grads = tape.gradient(max_prob, network.trainable_variables)
    assert all(not tf.reduce_all(tf.math.is_nan(g)).numpy() for g in grads)


def test_gcpn_actor_policy(tf_env):
    network = GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())
    actor = ActorPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        network
    )
    init_ts = tf_env.reset()

    # Test the deterministic output
    action = actor.action(init_ts)
    assert action.action.shape == (1, 2)

    # Test the probabilistic output
    dist = actor.distribution(init_ts)
    assert dist.action.mode().shape == (1, 2)

    # Make sure it is differentiable
    with tf.GradientTape() as tape:
        dist = actor.distribution(init_ts)
        max_prob = tf.reduce_max(dist.action.log_prob(dist.action.mode()))
    grads = tape.gradient(max_prob, network.trainable_variables)
    assert all(not tf.reduce_all(tf.math.is_nan(g)).numpy() for g in grads)


def test_gcpn_value_network(tf_env, example_batch):
    network = GCPNCriticNetwork(tf_env.observation_spec(), tf_env.reset())
    batch = convert_env_to_mpnn_batch(example_batch['observation'])
    network.create_variables()
    assert 'node_graph_indices' in batch
    assert batch['node_graph_indices'].numpy().max() == 1

    value_pred, _ = network.call(example_batch['observation'])
    assert value_pred.shape == (2,)

    # See if the network is differentiable
    with tf.GradientTape() as tape:
        value, _ = network.call(example_batch['observation'])
        max_value = tf.reduce_max(value)
    grads = tape.gradient(max_value, network.trainable_variables)
    assert all(not tf.reduce_all(tf.math.is_nan(g)).numpy() for g in grads)


def test_ppo_policy(tf_env, example_batch):
    actor_net = GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())
    critic_net = GCPNCriticNetwork(tf_env.observation_spec(), tf_env.reset())
    tf_agent = PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_net=actor_net,
        value_net=critic_net,
        optimizer=tf.keras.optimizers.Adam(1e-4),
        normalize_observations=False,
    )
    tf_agent.initialize()
    tf_agent.collect_policy.action(tf_env.reset())


def test_savedmodel(tmpdir, tf_env, example_batch):
    # Make a network
    actor_net = GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())

    # Save it to disk
    out_path = tmpdir / 'test'
    actor_net.save_as_savedmodel(out_path)

    # Load it in as a saved_model
    model = tf.saved_model.load(str(out_path))

    # Test it!
    observation = example_batch['observation']
    output = model.evaluate(
        observation['atom'],
        observation['bond'],
        observation['is_atom'],
        observation['connectivity'],
        observation['allowed_actions'],
    )
    assert tf.reduce_all(tf.equal(output, actor_net.compute_logits(observation)))
