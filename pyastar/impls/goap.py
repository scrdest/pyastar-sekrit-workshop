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
from pyastar.maputils import Map2D
from pyastar.measures import manhattan_distance, minkowski_distance, obstacle_dist


class EmptyQueueError(Exception):
    pass



def evaluate_neighbor(
    check_preconds,
    neigh,
    current_pos,
    goal,
    blackboard=None,
    measure=None,
    neighbor_measure=None,
    goal_measure=None
):
    _neighbor_measure = neighbor_measure or measure or manhattan_distance
    _goal_measure = goal_measure or measure or manhattan_distance
    _blackboard = blackboard or {}

    possible = check_preconds(neigh, _blackboard)

    if not possible:
        return float("inf")

    else:
        neigh_distance = _neighbor_measure(
            start=current_pos,
            end=neigh
        )

    goal_distance = _goal_measure(
        start=neigh,
        end=goal
    )

    heuristic = sum((
        neigh_distance,
        goal_distance
    ))

    # print(f"{current_pos}->{neigh} with cost {heuristic} ({neigh_distance}+{goal_distance})")

    return heuristic


def _astar_deepening_search(
    start_pos,
    goal,
    adjacency_gen,
    passability_checker,
    parent_callback=None,
    root=None,
    curr_cost=0,
    visited=None,
    paths=None,
    neighbor_measure=None,
    goal_measure=None,
    queue=False,
    blackboard=None,
):

    _root = root or start_pos
    _paths = paths or dict()
    _pqueue = queue or []
    _blackboard = blackboard or {}

    if start_pos == goal:
        cost, parent, stored_blackboard = _paths.get(goal) or (curr_cost, start_pos)

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
            check_preconds=passability_checker,
            neigh=neigh,
            current_pos=start_pos,
            goal=goal,
            neighbor_measure=_neighbor_measure,
            goal_measure=_goal_measure,
            blackboard=_blackboard,
        )

        stored_neigh_cost, stored_curr_parent, stored_blackboard = _paths.get(neigh) or (float("inf"), None, None)
        total_cost = curr_cost + heuristic

        if total_cost < stored_neigh_cost:
            _paths[neigh] = (total_cost, start_pos, None)

        heapq.heappush(
            _pqueue,
            (total_cost, neigh, None)
        )

    if not _pqueue:
        raise EmptyQueueError("Exhausted all candidates before a path was found!")

    cand_cost, cand_pos, cand_blackboard = heapq.heappop(_pqueue)

    return True, dict(
        start_pos=cand_pos,
        goal=goal,
        adjacency_gen=adjacency_gen,
        passability_checker=passability_checker,
        parent_callback=parent_callback,
        root=_root,
        curr_cost=cand_cost,
        visited=_visited,
        paths=_paths,
        neighbor_measure=_neighbor_measure,
        goal_measure=_goal_measure,
        queue=_pqueue,
    )


def solve_astar(
    start_pos,
    goal,
    adjacency_gen,
    passability_check,
    handle_backtrack_node,
    visited=None,
    paths=None,
    neighbor_measure=None,
    goal_measure=None,
):
    continue_search, next_params = True, dict(
        adjacency_gen=adjacency_gen,
        passability_checker=passability_check,
        parent_callback=handle_backtrack_node,
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths,
        neighbor_measure=neighbor_measure,
        goal_measure=goal_measure,
        blackboard=None,
    )

    best_cost, best_parent = None, None
    result_paths = dict()

    while next_params:
        continue_search, new_params = _astar_deepening_search(**next_params)
        last_params, next_params = next_params, new_params

        # print(next_params)
        if continue_search:
            result_paths = next_params["paths"]
        else:
            best_cost, best_parent = next_params
            break


    parent_cost, optimum_parent = best_cost, best_parent

    while optimum_parent is not start_pos:
        parent_cost, optimum_parent, stored_blackboard = result_paths.get(optimum_parent)
        handle_backtrack_node(optimum_parent)

    return best_cost, best_parent
