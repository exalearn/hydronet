from typing import Optional

from tf_agents.policies import TFPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types
import tensorflow as tf


# TODO (wardlt): Sutanay's advice on what to do next:
#  - Make a version of the environment that gives us an embedding (have the observations be *fingerprints*, etc)
#  - Explore categorical DQN with embedding features as a baseline
#  - Focus on making the baselines, scale-ability for the environments
#     - Note that Reverb (available on GitHub) can work with TF-Agents


class ConstrainedRandomPolicy(TFPolicy):
    """Policy that selects a random step from"""

    def _action(self, time_step: ts.TimeStep,
                policy_state: types.NestedTensor,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        # Get the adj matrix of possible actions
        assert time_step.observation['allowed_actions'].shape[0] == 1, f'We have not yet implemented batch_size > 1'
        allowed_actions_adj = time_step.observation['allowed_actions'][0, :, :]

        # Transform it to a list of tuples
        allowed_actions = tf.where(allowed_actions_adj != 0, name='find_allowed_actions')

        # Special case: No actions - Return a redo of the first bond (will not change graph
        if tf.size(allowed_actions) == 0:
            return policy_step.PolicyStep(
                tf.constant([[0, 1]], tf.int32),
                policy_state
            )

        # Pick one at random
        rid = tf.random.uniform(shape=[], maxval=tf.shape(allowed_actions)[0], dtype=tf.int32)
        return policy_step.PolicyStep(
            tf.cast([allowed_actions[rid]], tf.int32),
            policy_state
        )
