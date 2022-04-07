"""AStar algorithm.

Basic, for-loopy implementation.
"""

import heapq
from goapystar.map_2d.utils import Map2D, evaluate_neighbor, map_2
from goapystar.measures import minkowski_distance


def solve_astar(graph: Map2D, start_pos, goal, paths=None):
    pqueue = []
    _visited = set()
    _paths = paths or dict()

    _visited.add(start_pos)
    current_pos = start_pos

    while current_pos != goal:
        neighbors = graph.adjacent_lazy(current_pos)

        for neigh in neighbors:
            if neigh in _visited:
                continue

            heuristic = evaluate_neighbor(
                get_impassable=graph.is_impassable,
                neigh=neigh,
                current_pos=current_pos,
                goal=goal,
                measure=minkowski_distance(2)
            )

            stored_neigh_cost, stored_curr_parent = _paths.get(neigh) or (float("inf"), None)

            if heuristic < stored_neigh_cost:
                _paths[neigh] = (heuristic, current_pos)

            heapq.heappush(
                pqueue,
                (heuristic, neigh)
            )

        if not pqueue:
            raise LookupError("Exhausted all candidates before a path was found!")

        curr_cost, current_pos = heapq.heappop(pqueue)
        if current_pos in _visited:
            continue

        _visited.add(current_pos)

    return_curr = goal
    total_cost = None

    while return_curr != start_pos:
        graph.add_to_path(return_curr)
        cost, best_parent = _paths[return_curr]
        total_cost = cost if total_cost is None else total_cost + cost
        return_curr = best_parent

    print(f"TOTAL COST: {total_cost}")
    return graph


def main():
    start = (1, 1)
    goal = (0, 6)

    raw_map = map_2()

    newmap = (
        Map2D(raw_map=raw_map, diagonals=False)
        .set_current(start)
        .set_goal(goal)
        .visualize()
    )

    result = solve_astar(
        newmap,
        start_pos=start,
        goal=goal,
    )

    newmap.visualize()


if __name__ == '__main__':
    main()
