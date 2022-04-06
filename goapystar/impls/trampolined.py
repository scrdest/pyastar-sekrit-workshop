"""AStar algorithm.

Fancy recursive-ish implementation; instead of carrying a huge stack around,
we use a pure-ish function and pass *parameters* around
so that the outputs of one call serve as inputs to the next.

The core 'worker' function is wrapped in a 'manager' function that handles
scheduling, ferries parameters between calls, and handles the backtracking phase.

This is approximately Trampolining (in the Lisp sense) - we just return the parameters
to the function we need called rather than a proper thunk, because data is nicer to debug
and serde rather than dealing with raw continuations. It's not like we ever need to call
any *other* functions than the next worker call anyway.

More cacheable in principle AND resumable - since each worker call handles only a single
piece of work (evaluating one node), the full search can be chunked up arbitrarily.
"""

import heapq
from goapystar.maputils import Map2D, evaluate_neighbor, map_2
from goapystar.measures import minkowski_distance


# this is our 'worker' function
def _astar_deepening_search(graph: Map2D, start_pos, goal, root=None, curr_cost=0, visited=None, paths=None, measure=None, queue=None):
    _paths = paths or dict()

    if start_pos == goal:
        cost, parent = _paths.get(goal) or (curr_cost, start_pos)
        if parent:
            graph.add_to_path(parent)
            graph.visualize()
        return False, (cost, parent)

    _root = root or start_pos
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

    return True, dict(
        graph=graph,
        start_pos=cand_pos,
        goal=goal,
        curr_cost=cand_cost,
        root=_root,
        visited=_visited,
        paths=_paths,
        queue=_pqueue,
    )


# this is our 'trampoline' function
def solve_astar_deepening(graph: Map2D, start_pos, goal, visited=None, paths=None):
    continue_search, next_params = True, dict(
        graph=graph,
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths
    )
    best_cost, best_parent = None, None
    result_paths = dict()

    while next_params:
        continue_search, next_params = _astar_deepening_search(**next_params)
        print(next_params)
        if continue_search:
            result_paths = next_params["paths"]
        else:
            best_cost, best_parent = next_params
            break


    parent_cost, optimum_parent = best_cost, best_parent
    while optimum_parent is not start_pos:
        parent_cost, optimum_parent = result_paths.get(optimum_parent)
        graph.add_to_path(optimum_parent)

    return best_cost, best_parent


def main():
    start = (1, 1)
    goal = (3, 6)
    raw_map = map_2()

    newmap = (
        Map2D(raw_map=raw_map, diagonals=False)
        .set_current(start)
        .set_goal(goal)
        .visualize()
    )

    cost, path = solve_astar_deepening(
        newmap,
        start_pos=start,
        goal=goal
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
