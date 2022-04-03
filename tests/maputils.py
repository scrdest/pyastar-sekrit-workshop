from pyastar import maputils


def test_action_graph_simple_pos_get():
    testmap = maputils.reasoning_map()
    graph = maputils.ActionGraph(raw_map=testmap)
    position = graph["1"]
    assert position == {
        '2': 1,
        '3': 3
    }


def test_action_graph_compound_pos_get():
    testmap = maputils.reasoning_map()
    graph = maputils.ActionGraph(raw_map=testmap)
    position = graph["1,2"]
    assert position == {
        '1': 1,
        '3': 1
    }


def test_action_graph_adjacent_lazy():
    testmap = maputils.reasoning_map()
    graph = maputils.ActionGraph(raw_map=testmap)
    result = tuple(graph.adjacent_lazy("1"))
    assert result == ('2', '3')


def test_action_graph_adjacent():
    testmap = maputils.reasoning_map()
    graph = maputils.ActionGraph(raw_map=testmap)
    result = graph.adjacent("1")
    assert result == {'2', '3'}
