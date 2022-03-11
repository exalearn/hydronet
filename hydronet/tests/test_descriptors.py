
import networkx as nx

from hydronet.descriptors import count_rings


def test_count(triangle_cluster: nx.DiGraph):
    assert count_rings(triangle_cluster, 3) == 1
    assert count_rings(triangle_cluster, 4) == 0