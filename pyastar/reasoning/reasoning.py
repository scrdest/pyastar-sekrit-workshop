from pyastar.measures import action_graph_dist
from pyastar.maputils import ActionGraph
from pyastar.impls.goap import solve_astar


def main():
    start = "work"
    goal = "work again"

    raw_map = {
        "eat": {"sleep": 1, "wash dishes": 3},
        "sleep": {"wash dishes": 8, "eat": 1},
        "wash dishes": {"go to gym": 2, "exercise": 2, "work": 20, "work again": 20},
        "go to gym": {"sleep": 1, "exercise": 4},
        "exercise": {"work": 20, "work again": 20, "eat": 3},
        "work": {"sleep": 1, "eat": 5, "exercise": 20, "work again": 120}
    }

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=newmap.adjacent_lazy,
        passability_check=newmap.is_passable,
        handle_backtrack_node=lambda parent: newmap.add_to_path(parent),
        neighbor_measure=action_graph_dist(newmap),
        goal_measure=action_graph_dist(newmap, default_cost=0.)
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
