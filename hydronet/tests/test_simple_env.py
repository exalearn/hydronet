"""Test the OpenAI Gym environment"""
from pytest import fixture

from hydronet.rl.envs.gym import WaterCluster


@fixture
def env() -> WaterCluster:
    return WaterCluster()


def test_initialize(env: WaterCluster):
    assert env.reward() == 0
    assert env.num_steps_taken == 0


def test_stepping(env: WaterCluster):
    env.maximum_size = 3

    # Add a new water
    _, reward, done, _ = env.step((0, 1, False))
    assert reward == 1
    assert not done

    # Add a new water
    _, reward, done, _ = env.step((0, 2, False))
    assert reward == 2
    assert not done

    # Add another bond and stop
    state, reward, done, _ = env.step((1, 2, True))
    assert reward == 3
    assert done

    # Check to make sure the action space is valid
    assert state in env.observation_space
    assert env.action_space.get_donor_mask() == [False, True, True, True]  # 0 has two bonds
    assert env.action_space.get_acceptor_mask() == [True, True, False, True]  # 1 has two bonds
    move = env.action_space.sample()
    assert move in env.action_space
    assert 3 in move
