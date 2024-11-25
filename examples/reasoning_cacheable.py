from src.goapystar.maputils import load_map_json
from src.goapystar.impls.cacheable import cacheable_solver
from src.goapystar.usecases.actions import (
    get_actions,
    get_effects,
    neighbor_measure,
    preconds_checker_for,
    goal_checker_for
)
from src.goapystar.usecases.actiongraph.graph import ActionGraph
from src.goapystar.state import State
from src.goapystar.measures import no_goal_heuristic


def main():
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    goal = State.fromdict({"Rested": 10, "Debug": 1}, name="END")

    raw_map = load_map_json("map1")

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    find_plan = cacheable_solver(
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=newmap.add_to_path,
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=no_goal_heuristic,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        cutoff_iter=500,
        max_queue_size=100,
    )

    cost, path = find_plan(
        start_pos=start,
        goal=goal,
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
