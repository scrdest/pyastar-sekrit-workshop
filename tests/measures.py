from goapystar import measures
from goapystar.maputils import ActionGraph


def test_action_graph_dist():
    graph = ActionGraph()
    measurer = measures.action_graph_dist(graph)
    cost = measurer("1", "3")
    assert cost == 3

