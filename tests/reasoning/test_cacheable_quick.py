import pytest
from goapystar.reasoning.reasoning_cacheable import *


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

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Rested": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Rested": 10}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 400, 200),
))
def test_multigoal_found_complex_food_dirty(mapname, maxiters, maxheap):
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 200, 200),
))
def test_multigoal_found_complex_food_clean(mapname, maxiters, maxheap):
    start = State.fromdict({"HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Debug": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({"HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Money": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Money": 1}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Debug": 5}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Money": 50}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    import functools

    cached_solver = functools.lru_cache(maxsize=10)(solve_astar)

    cost, path = cached_solver(
        start_pos=start,
        goal=goal,
    )

    cost2, path2 = cached_solver(
        start_pos=start,
        goal=goal,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
    assert path2
    assert path2 is path  # proof caching worked


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("complex_sleepless", 500, 500),
    ("complex_nodebug", 700, 500),
    # ("complex_sleepless_workhard", 4500, 2000),  # fails - cycle, I think
))
def test_repeated_multigoal_moneyrest(mapname, maxiters, maxheap):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Money": 30, "Rested": 5}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
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
    start = State.fromdict({"HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 1, "Rested": 10, "Money": 10}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=maxiters, max_heap_size=maxheap)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path
