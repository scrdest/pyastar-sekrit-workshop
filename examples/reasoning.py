from goapystar.maputils import load_map_json

from goapystar.impls.goap import find_plan
from goapystar.usecases.actions import (
    get_actions,
    get_effects,
    neighbor_measure,
    preconds_checker_for,
    goal_checker_for
)
from goapystar.usecases.actiongraph.actiongraph import ActionGraph
from goapystar.measures import no_goal_heuristic


def custom_map():
    return load_map_json("map1")


def main():
    start = {"HasDirtyDishes": 1}
    goal = {"Debug": 1}

    raw_map = custom_map()

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
        cutoff_iter=500,
        max_queue_size=100,
        blackboard_default=0,
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
