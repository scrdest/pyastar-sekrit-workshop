import pytest
from pyastar.reasoning.reasoning import *


@pytest.mark.parametrize("mapname", (
    ("debug_only"),
    ("debug_complex"),
))
def test_debug_simple(mapname):
    start = State.fromdict({"Debug": 1}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=10
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 1
    assert path[0] == start



@pytest.mark.parametrize("mapname", (
    ("debug_only"),
    # ("debug_complex"),
))
def test_debug_found_simple(mapname):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 2
    assert path[0] == start
    assert path[1] == "DebugGetSimple"


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_debug_found_problematic(mapname):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=500
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 2
    assert path[0] == start
    assert path[1] == "DebugGetSimple"



@pytest.mark.parametrize("mapname", (
    ("fed_only"),
))
def test_fed_found_simple(mapname):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Fed": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=500
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 3
    assert path[0] == start
    assert path[1] == "GetFood"
    assert path[2] == "Eat"


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_fed_found_complex(mapname):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert len(path) == 6
    assert path[0] == start


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_smol_rested_found_complex(mapname):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Rested": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_big_rested_found_complex(mapname):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Rested": 10}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_multigoal_found_complex_food_dirty(mapname):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_multigoal_found_complex_food_clean(mapname):
    start = State.fromdict({"HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path




@pytest.mark.parametrize("mapname", (
    ("debug_complex"),
))
def test_multigoal_found_complex_food_clean(mapname):
    start = State.fromdict({"HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Money": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=1000
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path

