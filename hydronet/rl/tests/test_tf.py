"""Test the OpenAI Gym environment"""
from tf_agents.trajectories import time_step as ts
import tensorflow as tf
from pytest import fixture
import numpy as np

from hydronet.data import graph_is_valid
from hydronet.rl.envs.tf import SimpleEnvironment


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
    assert np.all(obs['atom'] == [0] * 11)
    assert np.all(obs['connectivity'][0, :] == [0, 1])


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
