from pytest import fixture

import networkx as nx


@fixture()
def triangle_cluster() -> nx.DiGraph:
    # Add three nodes
    g = nx.DiGraph()
    for i in range(3):
        g.add_node(i)

    # Make it so they are bonded in a circle
    for a in range(3):
        b = (a + 1) % 3
        g.add_edge(a, b, label='donate')
        g.add_edge(b, a, label='accept')
    return g
