"""AStar algorithm.

Basic recursive implementation.
More cacheable in principle, but will stack-overflow on larger maps.
"""

import heapq
from goapystar.map_2d.utils import Map2D, evaluate_neighbor, map_2
from goapystar.measures import minkowski_distance


def _astar_recursive_search(graph: Map2D, start_pos, goal, root=None, curr_cost=0, visited=None, paths=None, measure=None, queue=None):
    _root = root or start_pos
    _paths = paths or dict()

    if start_pos == goal:
        cost, parent = _paths.get(goal) or (curr_cost, start_pos)
        if parent:
            graph.add_to_path(parent)
            graph.visualize()
        return cost, parent

    _pqueue = queue or []
    _measure = measure or minkowski_distance(2)
    _visited = visited or set()
    _visited.add(start_pos)
    neighbors = graph.adjacent_lazy(start_pos)

    for neigh in neighbors:
        if neigh in _visited:
            continue

        heuristic = evaluate_neighbor(
            get_impassable=graph.is_impassable,
            neigh=neigh,
            current_pos=start_pos,
            goal=goal,
            measure=_measure
        )

        stored_neigh_cost, stored_curr_parent = _paths.get(neigh) or (float("inf"), None)
        total_cost = curr_cost + heuristic

        if total_cost < stored_neigh_cost:
            _paths[neigh] = (total_cost, start_pos)

        heapq.heappush(
            _pqueue,
            (total_cost, neigh)
        )

    if not _pqueue:
        raise LookupError("Exhausted all candidates before a path was found!")

    cand_cost, cand_pos = heapq.heappop(_pqueue)

    # could return just the parameters for deeper search...
    best_cost, best_parent = _astar_recursive_search(
        graph=graph,
        start_pos=cand_pos,
        goal=goal,
        curr_cost=cand_cost,
        root=_root,
        visited=_visited,
        paths=_paths,
        queue=_pqueue,
    )

    parent_cost, optimum_parent = _paths.get(best_parent) or (curr_cost, _root)
    graph.add_to_path(optimum_parent)

    return best_cost, optimum_parent


def solve_astar_recursive(graph: Map2D, start_pos, goal, visited=None, paths=None):
    best_cost, best_path = _astar_recursive_search(
        graph=graph,
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths
    )
    return best_cost, "->".join(map(str, (start_pos, best_path)))


def main():
    start = (1, 1)
    goal = (9, 0)

    raw_map = map_2()

    newmap = (
        Map2D(raw_map=raw_map, diagonals=False)
        .set_current(start)
        .set_goal(goal)
        .visualize()
    )

    cost, path = solve_astar_recursive(
        newmap,
        start_pos=start,
        goal=goal
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
