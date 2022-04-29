from pytest import fixture
from tf_agents.environments import TFPyEnvironment

from hydronet.workflow import train_rl_policy
from hydronet.rl.tf.env import SimpleEnvironment
from hydronet.rl.tf.networks import GCPNActorNetwork, GCPNCriticNetwork


@fixture()
def env():
    return SimpleEnvironment()


@fixture()
def tf_env(env):
    return TFPyEnvironment(env)


@fixture()
def actor_net(tf_env):
    return GCPNActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), tf_env.reset())


@fixture()
def critic_net(tf_env):
    return GCPNCriticNetwork(tf_env.observation_spec(), tf_env.reset())


def test_train_rl(env, actor_net, critic_net):
    actor_net, critic_net, train_log = train_rl_policy(
        env=env,
        actor_net=actor_net,
        critic_net=critic_net,
        training_cycles=2,
        buffer_size=32,
        episodes_per_cycle=4,
    )
    assert 'loss' in train_log
