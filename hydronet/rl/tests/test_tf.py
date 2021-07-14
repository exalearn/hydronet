"""Test the OpenAI Gym environment"""
from tf_agents.trajectories import time_step as ts
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
    assert len(env.time_step_spec()) == 4  # Just make sure it has 4 entries and builds correctly
    assert env.action_spec().shape == (2,)


def test_stepping(env: SimpleEnvironment):
    env.maximum_size = 3

    # Add a new water
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