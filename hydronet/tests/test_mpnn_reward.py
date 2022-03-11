from math import isclose
import pickle as pkl
import os

from tensorflow.keras.models import load_model
from pytest import fixture

from hydronet.rl.rewards.mpnn import MPNNReward
from hydronet.mpnn.layers import custom_objects


_home_dir = os.path.dirname(__file__)


@fixture
def model():
    return load_model(os.path.join(_home_dir, 'model.h5'),
                      custom_objects=custom_objects)


def test_mpnn_reward(model, triangle_cluster):
    reward = MPNNReward(model)
    assert isinstance(reward(triangle_cluster), float)


def test_pickle(model, triangle_cluster):
    # Run inference on the first graph
    reward = MPNNReward(model)
    reward(triangle_cluster)

    # Clone the model
    reward2 = pkl.loads(pkl.dumps(reward))

    assert isclose(reward(triangle_cluster), reward2(triangle_cluster), abs_tol=1e-6)
