import pytest
from examples.reasoning import *
from goapystar.usecases.actiongraph.actiongraph import ActionGraph
from goapystar.maputils import load_map_json
from goapystar.state import State


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_only", 10, 10),
    ("debug_complex", 10, 50),
    ("debug_complex", 10, None),
    ("debug_complex", None, 50),
    ("debug_complex", None, None),
))
def test_debug_startisend(mapname, maxiters, maxheap):
    start = State.fromdict({"Debug": 1}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 1
    assert path[0] == start



@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_only", 10, 20),
    ("debug_only", None, 20),
    ("debug_only", 10, None),
    ("debug_only", None, None),
))
def test_debug_found_simple(mapname, maxiters, maxheap):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 2
    assert path[0] == start
    assert path[1] == "DebugGetSimple"


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 1000, 500),
))
def test_debug_found_problematic(mapname, maxiters, maxheap):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 2
    assert path[0] == start
    assert path[1] == "DebugGetSimple"



@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("fed_only", 100, 200),
))
def test_fed_found_simple(mapname, maxiters, maxheap):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Fed": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 3
    assert path[0] == start
    assert path[1] == "GetFood"
    assert path[2] == "Eat"


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 200, 200),
    ("complex_nodebug", 100, 200),
))
def test_fed_found_complex(mapname, maxiters, maxheap):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 6
    assert path[0] == start


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 10, 200),
    ("complex_nodebug", 10, 200),
))
def test_smol_rested_found_complex(mapname, maxiters, maxheap):
    start = {"HasDirtyDishes": 1}
    goal = {"Rested": 1}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 400, 200),
    ("complex_nodebug", 300, 200),
))
def test_big_rested_found_complex(mapname, maxiters, maxheap):
    start = {"HasDirtyDishes": 1}
    goal = {"Rested": 10}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 400, 200),
))
def test_multigoal_found_complex_food_dirty(mapname, maxiters, maxheap):
    start = {"HasDirtyDishes": 1}
    goal = {"Fed": 1, "Debug": 1}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 200, 200),
))
def test_multigoal_found_complex_food_clean(mapname, maxiters, maxheap):
    start = {"HasCleanDishes": 1}
    goal = {"Fed": 1, "Debug": 1}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 1000, 200),
    ("complex_nodebug", 1000, 200),
))
def test_multigoal_found_complex_foodmoney_clean(mapname, maxiters, maxheap):
    start = {"HasCleanDishes": 1}
    goal = {"Fed": 1, "Money": 1}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 500, 200),
    ("complex_nodebug", 300, 200),
    ("complex_sleepless", 300, 200),
))
def test_multigoal_found_complex_foodmoney_clean(mapname, maxiters, maxheap):
    start = {"HasDirtyDishes": 1}
    goal = {"Fed": 1, "Money": 1}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path



@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_only", 10, 5000),
    ("debug_complex", 300, 5000),
))
def test_repeated_goal_debug(mapname, maxiters, maxheap):
    start = {}
    goal = {"Debug": 5}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("complex_sleepless", 200, 200),
    ("debug_complex", 600, 450),
    ("complex_nodebug", 200, 200),
    ("complex_sleepless_workhard", 1500, 1000),
))
def test_repeated_goal_money(mapname, maxiters, maxheap):
    start = {}
    goal = {"Money": 50}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("complex_sleepless", 500, 500),
    ("complex_nodebug", 700, 500),
    # ("complex_sleepless_workhard", 4500, 2000),  # fails - cycle, I think
))
def test_repeated_multigoal_moneyrest(mapname, maxiters, maxheap):
    start = {}
    goal = {"Money": 30, "Rested": 5}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()


    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 600, 5000),
    ("complex_nodebug_workhard", 300, 5000),
))
def test_repeated_multigoal_foodrestmoney(mapname, maxiters, maxheap):
    start = {"HasCleanDishes": 1}
    goal = {"Fed": 1, "Rested": 10, "Money": 10}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("custom_binop_test1", 50, 100),
))
def test_custom_updateop_modes(mapname, maxiters, maxheap):
    start = {"ReadMode": True, "IsTrue": False}
    goal = {"IsTrue": True, "ReadMode": True}

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    def custom_bin_op(src_val, new_val):
        result = new_val
        return result

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map, cmp_op=lambda x, y: x != y),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
        blackboard_update_op=custom_bin_op,
        blackboard_default=0,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("diagonals", "maxiters", "maxheap"), (
    (False, 5000, 10000),
    (True, 10000, 10000),
))
def test_custom_updateop_maplike(diagonals, maxiters, maxheap):
    start = {"pos": (1, 1)}
    goal = {"pos": (19, 2)}

    from goapystar.usecases.map_2d.utils import map_3, Map2D

    raw_map = map_3()

    newmap = (
        Map2D(raw_map, diagonals=diagonals)
        .set_current(start["pos"])
        .set_goal(goal["pos"])
    )

    def custom_bin_op(src_val, new_val):
        result = new_val
        return result

    def custom_heuristic(src, trg):
        from goapystar.measures import manhattan_distance
        src_pos = src["pos"] if isinstance(src, (dict, State)) else src
        trg_pos = trg["pos"] if isinstance(trg, (dict, State)) else trg
        return manhattan_distance(src_pos, trg_pos)

    def adjacents(pos):
        curr_pos = pos["pos"] if isinstance(pos, (dict, State)) else pos
        raw_result = newmap.adjacent(curr_pos)
        result = raw_result
        return result

    def new_pos(pos):
        curr_pos = pos
        result = {"pos": curr_pos}
        return result

    def is_passable(pos, state):
        curr_pos = pos
        result = newmap.is_passable(curr_pos)
        return result

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
        adjacency_gen=adjacents,
        preconditions_check=is_passable,
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=custom_heuristic,
        goal_measure=custom_heuristic,
        goal_check=goal_checker_for(raw_map, cmp_op=lambda x, y: x != y),
        get_effects=new_pos,
        visited=dict(),
        cutoff_iter=maxiters,
        max_queue_size=maxheap,
        blackboard_update_op=custom_bin_op,
        blackboard_default=0,
        pqueue_key_func=lambda _iter, curr_cost, heuristic: (1,)
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
