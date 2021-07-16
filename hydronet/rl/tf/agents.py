from typing import Optional

from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types
import numpy as np


class RandomPolicy(PyPolicy):

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedArray,
                seed: Optional[types.Seed] = None) -> policy_step.PolicyStep:
        # Get the adj matrix of possible actions
        allowed_actions_adj = time_step.observation['allowed_actions']

        # Transform it to a list of tuples
        allowed_actions = np.transpose(np.nonzero(allowed_actions_adj))

        # Pick one at random
        rng = np.random.RandomState(seed)
        rid = rng.choice(len(allowed_actions))
        return policy_step.PolicyStep(
            np.array(allowed_actions[rid], np.int32),
            policy_state
        )
