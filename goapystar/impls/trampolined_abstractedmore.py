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
from goapystar.maputils import Map2D, evaluate_neighbor, map_3
from goapystar.measures import manhattan_distance, minkowski_distance, obstacle_dist


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
    neighbor_measure=None,
    goal_measure=None,
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

    _neighbor_measure = neighbor_measure or minkowski_distance(2)
    _goal_measure = goal_measure or minkowski_distance(2)
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
            neighbor_measure=_neighbor_measure,
            goal_measure=_goal_measure,
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
        neighbor_measure=_neighbor_measure,
        goal_measure=_goal_measure,
        queue=_pqueue,
    )


def solve_astar_deepening(
    start_pos,
    goal,
    adjacency_gen,
    impassability_check,
    handle_backtrack_node,
    visited=None,
    paths=None,
    neighbor_measure=None,
    goal_measure=None,
):
    continue_search, next_params = True, dict(
        adjacency_gen=adjacency_gen,
        impassability_checker=impassability_check,
        parent_callback=handle_backtrack_node,
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths,
        neighbor_measure=neighbor_measure,
        goal_measure=goal_measure,
    )
    best_cost, best_parent = None, None
    result_paths = dict()

    while next_params:
        try:
            continue_search, new_params = _astar_deepening_search(**next_params)
            last_params, next_params = next_params, new_params

        except ModuleNotFoundError:
            continue_search, next_params = False, next_params
        # print(next_params)
        if continue_search:
            result_paths = next_params["paths"]
        else:
            best_cost, best_parent = next_params
            break


    parent_cost, optimum_parent = best_cost, best_parent
    while optimum_parent is not start_pos:
        parent_cost, optimum_parent = result_paths.get(optimum_parent)
        handle_backtrack_node(optimum_parent)

    return best_cost, best_parent


def main():
    start = (1, 1)
    goal = (19, 2)

    raw_map = map_3()

    newmap = (
        Map2D(raw_map=raw_map, diagonals=False)
        .set_current(start)
        .set_goal(goal)
        .visualize()
    )

    cost, path = solve_astar_deepening(
        start_pos=start,
        goal=goal,
        adjacency_gen=newmap.adjacent_lazy,
        impassability_check=newmap.is_impassable,
        handle_backtrack_node=lambda parent: newmap.add_to_path(parent),
        neighbor_measure=obstacle_dist(newmap, manhattan_distance),
        goal_measure=obstacle_dist(newmap, manhattan_distance),
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
