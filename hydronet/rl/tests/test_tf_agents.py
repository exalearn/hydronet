import numpy as np
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.policies import ActorPolicy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs.tensor_spec import add_outer_dim
import tensorflow as tf

from pytest import fixture
from tf_agents.typing.types import NestedTensor

from hydronet.rl.tf.networks import GCPN
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
        time_step_spec=add_outer_dim(tf_env.time_step_spec(), tf_env.batch_size),
        action_spec=add_outer_dim(tf_env.action_spec(), tf_env.batch_size)
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


def test_gcpn_network(tf_env, example_batch):
    network = GCPN(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())
    batch = network.convert_env_to_mpnn_batch(example_batch['observation'])
    assert 'node_graph_indices' in batch
    assert batch['node_graph_indices'].numpy().max() == 1

    action_choice, _ = network.call(example_batch['observation'])
    assert action_choice.sample().shape == (2, 2)


def test_gcpn_policy(tf_env):
    network = GCPN(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())
    actor = ActorPolicy(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        network
    )
    init_ts = tf_env.reset()
    action = actor.action(init_ts)
    print(action)
