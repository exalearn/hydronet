
# TODO (wardlt): Switch to ExaRL or TF-Agents spec for agents
from hydronet.rl.envs.simple import WaterCluster


class BaseAgent:
    """A base class for agents."""

    def action(self, env: WaterCluster):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Pick moves randomly"""

    def action(self, env: WaterCluster):
        return env.action_space.sample()
