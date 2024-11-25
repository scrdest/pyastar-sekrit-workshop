from src import goapystar
import src.goapystar.usecases.actiongraph.utils


def test_action_graph_simple_pos_get():
    testmap = src.goapystar.usecases.actiongraph.utils.reasoning_map()
    graph = src.goapystar.usecases.actiongraph.graph.ActionGraph(raw_map=testmap)
    position = graph["1"]
    assert position == {
        '2': 1,
        '3': 3
    }


def test_action_graph_compound_pos_get():
    testmap = src.goapystar.usecases.actiongraph.utils.reasoning_map()
    graph = src.goapystar.usecases.actiongraph.graph.ActionGraph(raw_map=testmap)
    position = graph["1,2"]
    assert position == 1


def test_action_graph_adjacent_lazy():
    testmap = src.goapystar.usecases.actiongraph.utils.reasoning_map()
    graph = src.goapystar.usecases.actiongraph.graph.ActionGraph(raw_map=testmap)
    result = tuple(graph.adjacent_lazy("1"))
    assert result == ('2', '3')


def test_action_graph_adjacent():
    testmap = src.goapystar.usecases.actiongraph.utils.reasoning_map()
    graph = src.goapystar.usecases.actiongraph.graph.ActionGraph(raw_map=testmap)
    result = graph.adjacent("1")
    assert result == {'2', '3'}
