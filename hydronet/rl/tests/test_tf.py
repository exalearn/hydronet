"""Test the OpenAI Gym environment"""
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from pytest import fixture
import numpy as np

from hydronet.data import graph_is_valid
from hydronet.rl.tf.env import SimpleEnvironment


@fixture
def env() -> SimpleEnvironment:
    env = SimpleEnvironment()
    env.reset()
    return env


def test_initialize(env: SimpleEnvironment):
    assert graph_is_valid(env.get_state(), coarse=True)
    assert env.current_time_step().step_type == ts.StepType.FIRST

    # Test the observation space
    obs = env.current_time_step().observation
    assert obs['n_atoms'] == 2
    assert np.all(obs['atom'] == [0] * 12)
    assert np.all(obs['connectivity'][0, :] == [0, 1])
    assert np.all(obs['connectivity'][-1, :] == [11, 11])


def test_specs(env: SimpleEnvironment):
    # Just make sure it has 4 entries and builds correctly
    assert len(env.time_step_spec()) == 4
    assert env.action_spec().shape == (2,)


def test_states(env: SimpleEnvironment):
    init_ts = env.get_state_as_tensors()
    for k, v in init_ts.items():
        assert isinstance(v, tf.Tensor), f'{k} is not a Tensor'


def test_stepping(env: SimpleEnvironment):
    env.maximum_size = 3

    # Re-add a bond
    step = env.step((0, 1))
    assert graph_is_valid(env.get_state(), coarse=True)
    assert step.reward == 1
    assert step.step_type == ts.StepType.MID

    # Add a new water
    step = env.step((0, 2))
    assert graph_is_valid(env.get_state(), coarse=True)
    assert step.reward == 2
    assert step.step_type == ts.StepType.MID

    # Add another bond
    step = env.step((1, 2))
    assert graph_is_valid(env.get_state(), coarse=True)
    assert step.reward == 3
    assert step.step_type == ts.StepType.MID

    # Add another water, which stops the episode
    step = env.step((3, 2))
    assert graph_is_valid(env.get_state(), coarse=True)
    assert step.reward == 4
    assert step.step_type == ts.StepType.LAST

    # Add an invalid step, make sure it kills the episode
    env.reset()
    assert graph_is_valid(env.get_state(), coarse=True)

    step = env.step((0, 0))
    assert not graph_is_valid(env.get_state(), coarse=True)
    assert step.reward == 0
    assert step.step_type == ts.StepType.LAST

    # Test a move that creates a second water cluster
    env.reset()
    assert graph_is_valid(env.get_state(), coarse=True)

    step = env.step((2, 3))
    assert step.step_type == ts.StepType.LAST
    assert step.reward == 0

    # Flip a bond, which is OK as long as it doesn't break other rules
    env.reset()

    step = env.step((1, 0))
    assert step.step_type == ts.StepType.MID
    assert step.reward == 1


def test_valid_moves(env: SimpleEnvironment):
    env.maximum_size = 3

    # Starting from two bonded waters, the only action is to add a new water
    valid_moves = env.get_valid_moves()
    assert set(valid_moves) == {(0, 2), (1, 2), (2, 0), (2, 1)}

    # After adding that third water, you can connect it to the second water or add a fourth
    #  We take a step such that the second water is accepting 2 bonds, so it can no-longer donate
    env.step((2, 1))
    valid_moves = env.get_valid_moves()
    assert set(valid_moves) == {(0, 2), (0, 3), (1, 3), (2, 0), (2, 3), (3, 0), (3, 2)}

    # After completing the triangle, the third water has now donated twice. It may donate no further
    env.step((2, 0))
    valid_moves = env.get_valid_moves()
    assert set(valid_moves) == {(0, 3), (1, 3), (3, 0), (3, 2)}

    # Test that last one in matrix form
    valid_moves = env.get_valid_actions_as_matrix()
    assert valid_moves.sum() == 4  # There are only 4 possible moves
    assert valid_moves[0, :].sum() == 1  # First water can only donate to a new one
    assert valid_moves[:, 0].sum() == 1  # First water can only accept from a new one
    assert valid_moves[0, 3] == 1  # Checking a specific move

    # When you add the last water and exceed the maximum size, you may add no more
    obs = env.step((3, 0))
    assert obs.step_type == ts.StepType.LAST
    valid_moves = env.get_valid_moves()
    assert not any(4 in x for x in valid_moves)  # Make sure we cannot add another water
