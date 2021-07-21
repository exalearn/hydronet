import numpy as np
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.specs.tensor_spec import add_outer_dim

from hydronet.rl.tf.agents import ConstrainedRandomPolicy
from hydronet.rl.tf.env import SimpleEnvironment


def test_random():
    env = SimpleEnvironment()
    tf_env = TFPyEnvironment(env)
    policy = ConstrainedRandomPolicy(
        time_step_spec=add_outer_dim(env.time_step_spec(), tf_env.batch_size),
        action_spec=add_outer_dim(env.action_spec(), tf_env.batch_size)
    )

    # Make an initial step
    init_step = tf_env.reset()
    allowed_adj = env.get_valid_actions_as_matrix()
    action = policy.action(init_step)
    action = np.squeeze(action.action)
    assert allowed_adj[action[0], action[1]] == 1


def test_with_driver():
    env = SimpleEnvironment()
    tf_env = TFPyEnvironment(env, check_dims=True)
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
