from hydronet.rl.tf.agents import RandomPolicy
from hydronet.rl.tf.env import SimpleEnvironment


def test_random():
    env = SimpleEnvironment()
    policy = RandomPolicy(
        time_step_spec=env.time_step_spec(),
        action_spec=env.action_spec()
    )

    # Make an initial step
    init_step = env.reset()
    allowed_adj = env.get_valid_actions_as_matrix()
    action = policy.action(init_step)
    assert allowed_adj[action.action[0], action.action[1]] == 1
