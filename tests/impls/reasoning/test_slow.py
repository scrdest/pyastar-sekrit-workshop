import pytest
from pyastar.reasoning.reasoning import *


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 4000, 3000),
    ("debug_complex", 4000, 5000),
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
        cutoff_iter=maxiters,
        max_heap_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()


    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("debug_complex", 4000, 3000),
))
def test_repeated_multigoal_moneyrestdebug(mapname, maxiters, maxheap):
    start = State.fromdict({}, name="START")
    goal = State.fromdict({"Money": 20, "Rested": 4, "Debug": 2}, name="END")

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
        cutoff_iter=maxiters,
        max_heap_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("complex_nodebug_ezwash", 17000, 5000),
))
def test_repeated_multigoal_customheuristic(mapname, maxiters, maxheap):
    # This is a deceptively hard problem - doesn't pass without the heuristic!
    start = State.fromdict({"Money": 200, "HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 3}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    def dict_measure(start, end):
        start_vals = get_effects(raw_map)(start) if isinstance(start, str) else start
        score = 0

        for end_key, end_val in end.items():
            if end_key not in start_vals.keys():
                score += max(10, end_val ** 4)

            else:
                start_val = start_vals[end_key]
                score += (end_val - start_val) ** 4

        return score

    print("")

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=dict_measure,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_heap_size=maxheap,
        # Oddly, the custom key format seems to be required ON TOP of the heuristic to finish reasonably fast
        pqueue_key_func=lambda it, co, he: (he, it)
    )

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

    def randomized_cost(start, end):
        import random
        return random.randint(-5, 5)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=randomized_cost,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_heap_size=maxheap,
    )

    print("")
    print("COST:", cost)

    newmap.visualize()

    assert path


@pytest.mark.parametrize(("mapname", "maxiters", "maxheap"), (
    ("complex_nodebug_ezwash", 17000, 5000),
))
def test_repeated_multigoal_randheuristic(mapname, maxiters, maxheap):
    # This is a deceptively hard problem - doesn't pass without the heuristic!
    start = State.fromdict({"Money": 200, "HasCleanDishes": 1}, name="START")
    goal = State.fromdict({"Fed": 3}, name="END")

    raw_map = load_map_json(mapname)

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    print("")

    def randomized_cost(start, end):
        # Oddly, the random-cost heuristic helps here *quite* a bit
        # the other run with dict measure runs consistent 16k+ iterations
        # -15, 15 range has gone as low as 13546 iters *so far*
        # -20, 20 range has gone as low as 12507 iters *so far*
        # -30, 30 range has gone as low as 11626 iters *so far*
        import random
        return random.randint(-30, 30)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=randomized_cost,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=maxiters,
        max_heap_size=maxheap,
    )

    print("COST:", cost)

    newmap.visualize()

    assert path