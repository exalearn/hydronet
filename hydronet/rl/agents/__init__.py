
# TODO (wardlt): Switch to ExaRL or TF-Agents spec for agents
from hydronet.rl.envs.gym import WaterCluster


class BaseAgent:
    """A base class for agents."""

    def action(self, env: WaterCluster):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Pick moves randomly, never stop explicitly"""

    def action(self, env: WaterCluster):
        return list(env.action_space.sample())
