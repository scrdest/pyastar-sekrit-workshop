from src.goapystar import measures
from src.goapystar.usecases.actiongraph.graph import ActionGraph


def test_action_graph_dist():
    graph = ActionGraph()
    measurer = measures.action_graph_dist(graph)
    cost = measurer("1", "3")
    assert cost == 3

