"""Different choices for reward functions"""

import networkx as nx


class RewardFunction:
    """Base class for water cluster reward functions"""

    def __init__(self, maximize: bool = True):
        """
        Args:
            maximize (bool): Whether to maximize the objective function
        """
        self.maximize = maximize

    def __call__(self, graph: nx.DiGraph) -> float:
        """Compute the reward for a certain molecule

        Args:
            graph (str): NetworkX graph form of the molecule
        Returns:
            (float) Reward
        """
        reward = self._call(graph)
        if self.maximize:
            return reward
        return -1 * reward

    def _call(self, graph: nx.DiGraph) -> float:
        """Compute the reward for a certain molecule

        Private version of the method. The public version
        handles switching signs if needed

        Args:
            graph (str): NetworkX graph form of the molecule
        Returns:
            (float) Reward
        """
        raise NotImplementedError()


class BondCountReward(RewardFunction):
    """A reward where we seek to maximize the number of bonds"""

    maximize = True

    def _call(self, graph: nx.DiGraph) -> float:
        return len(graph.edges) / 2.
