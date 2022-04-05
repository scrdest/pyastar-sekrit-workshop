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

This is a variant of the 'trampolined' module. Most of the references to custom data structures
and stateful nonsense had been quarantined behind callbacks to make things more generic.
"""


import heapq
from goapystar.maputils import Map2D, evaluate_neighbor
from goapystar.measures import minkowski_distance


class EmptyQueueError(Exception):
    pass


def _astar_deepening_search(
    start_pos,
    goal,
    adjacency_gen,
    impassability_checker,
    parent_callback=None,
    root=None,
    curr_cost=0,
    visited=None,
    paths=None,
    measure=None,
    queue=False,
):

    _root = root or start_pos
    _paths = paths or dict()
    _pqueue = queue or []

    if start_pos == goal:
        cost, parent = _paths.get(goal) or (curr_cost, start_pos)
        if parent and parent_callback:
            parent_callback(parent)
        return False, (cost, parent)

    _measure = measure or minkowski_distance(2)
    _visited = visited or set()
    _visited.add(start_pos)
    neighbors = adjacency_gen(start_pos)

    for neigh in neighbors:
        if neigh in _visited:
            continue

        heuristic = evaluate_neighbor(
            get_impassable=impassability_checker,
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
        raise EmptyQueueError("Exhausted all candidates before a path was found!")

    cand_cost, cand_pos = heapq.heappop(_pqueue)

    return True, dict(
        start_pos=cand_pos,
        goal=goal,
        adjacency_gen=adjacency_gen,
        impassability_checker=impassability_checker,
        parent_callback=parent_callback,
        root=_root,
        curr_cost=cand_cost,
        visited=_visited,
        paths=_paths,
        measure=measure,
        queue=_pqueue,
    )


def solve_astar_deepening(graph: Map2D, start_pos, goal, visited=None, paths=None, measure=None):
    continue_search, next_params = True, dict(
        adjacency_gen=graph.adjacent_lazy,
        impassability_checker=graph.is_passable,
        parent_callback=lambda parent: graph.add_to_path(parent) and graph.visualize(),
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths,
        measure=measure,
    )
    best_cost, best_parent = None, None
    result_paths = dict()

    while next_params:
        try:
            continue_search, new_params = _astar_deepening_search(**next_params)
            last_params, next_params = next_params, new_params

        except ModuleNotFoundError:
            continue_search, next_params = False, next_params
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
    goal = (19, 2)

    newmap = (
        Map2D(diagonals=False)
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
