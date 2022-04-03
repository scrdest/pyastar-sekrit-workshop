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
import copy
import heapq

from pyastar.measures import action_graph_dist


class EmptyQueueError(Exception):
    pass


class NoPathError(Exception):
    pass


PLUS_INF = float("inf")
BLACKBOARD_CLASS = dict


def evaluate_neighbor(
    check_preconds,
    neigh,
    current_pos,
    goal,
    blackboard=None,
    measure=None,
    neighbor_measure=None,
    goal_measure=None,
    get_effects=None,
):
    _neighbor_measure = neighbor_measure or measure or action_graph_dist
    _goal_measure = goal_measure or measure or action_graph_dist
    _blackboard = blackboard or BLACKBOARD_CLASS()

    valid = check_preconds(neigh, _blackboard)

    effects = copy.deepcopy(_blackboard or BLACKBOARD_CLASS())
    effects.setdefault("src", []).append(current_pos)
    if get_effects:
        new_effects = get_effects(neigh)
        for state_key, state_val in new_effects.items():
            effects[state_key] = effects.get(state_key, 0) + state_val

    if not valid:
        return PLUS_INF, effects

    neigh_distance = _neighbor_measure(
        current_pos,
        neigh
    )

    goal_distance = _goal_measure(
        neigh,
        goal
    )

    heuristic = sum((
        neigh_distance,
        goal_distance
    ))

    return heuristic, effects


def equality_check(pos, goal):
    return pos == goal


def _astar_deepening_search(
    start_pos,
    goal,
    adjacency_gen,
    preconditions_checker,
    parent_callback=None,
    curr_cost=0,
    visited=None,
    paths=None,
    neighbor_measure=None,
    goal_measure=None,
    queue=False,
    goal_checker=None,
    get_effects=None,
    blackboard=None,
    _iter=1
):

    _paths = paths or dict()
    _pqueue = queue or []
    _blackboard = blackboard or BLACKBOARD_CLASS()
    _goal_check = goal_checker or equality_check

    if _goal_check(_blackboard, goal):
        cost, parent, _, _ = _paths.get(goal) or (curr_cost, start_pos, None, None)

        if parent and parent_callback:
            parent_callback(parent)

        return False, (cost, parent)

    _neighbor_measure = neighbor_measure or action_graph_dist
    _goal_measure = goal_measure or action_graph_dist
    _visited = visited or dict()

    if start_pos not in _visited:
        _visited[start_pos] = _visited.get(start_pos, 0) + 1

    neighbors = adjacency_gen(start_pos)

    for neigh in neighbors:
        if _visited.get(neigh, 0) > 1:
            continue

        heuristic, effects = evaluate_neighbor(
            check_preconds=preconditions_checker,
            neigh=neigh,
            current_pos=start_pos,
            goal=goal,
            neighbor_measure=_neighbor_measure,
            goal_measure=_goal_measure,
            blackboard=_blackboard,
            get_effects=get_effects
        )

        stored_neigh_cost, stored_curr_parent, _ = _paths.get(neigh) or (PLUS_INF, None, None)
        total_cost = curr_cost + heuristic
        frozen_effects = tuple(effects.items())

        if total_cost < stored_neigh_cost:
            _paths[neigh] = (total_cost, start_pos, frozen_effects)

        cand_tuple = (total_cost, neigh, start_pos, frozen_effects)

        if cand_tuple not in _pqueue and total_cost < PLUS_INF:
            heapq.heappush(
                _pqueue,
                cand_tuple
            )

    if not _pqueue:
        raise EmptyQueueError("Exhausted all candidates before a path was found!")

    try:
        cand_cost, cand_pos, src_pos, raw_cand_blackboard = heapq.heappop(_pqueue)

    except TypeError as Terr:
        print(_pqueue)
        raise

    cand_blackboard = dict(raw_cand_blackboard)

    result = True, dict(
        start_pos=cand_pos,
        goal=goal,
        adjacency_gen=adjacency_gen,
        preconditions_checker=preconditions_checker,
        parent_callback=parent_callback,
        curr_cost=cand_cost,
        visited=_visited,
        paths=_paths,
        neighbor_measure=_neighbor_measure,
        goal_measure=_goal_measure,
        queue=_pqueue,
        goal_checker=_goal_check,
        get_effects=get_effects,
        blackboard=cand_blackboard,
        _iter=_iter+1
    )
    return result


def solve_astar(
    start_pos,
    goal,
    adjacency_gen,
    preconditions_check,
    handle_backtrack_node,
    visited=None,
    paths=None,
    neighbor_measure=None,
    goal_measure=None,
    blackboard=None,
    goal_check=None,
    get_effects=None,
):
    continue_search, next_params = True, dict(
        adjacency_gen=adjacency_gen,
        preconditions_checker=preconditions_check,
        parent_callback=handle_backtrack_node,
        start_pos=start_pos,
        goal=goal,
        visited=visited,
        paths=paths,
        neighbor_measure=neighbor_measure,
        goal_measure=goal_measure,
        blackboard=blackboard,
        goal_checker=goal_check,
        get_effects=get_effects,
    )

    best_cost, best_parent = None, None
    result_paths = dict()

    while next_params:
        continue_search, new_params = _astar_deepening_search(**next_params)
        last_params, next_params = next_params, new_params

        if continue_search:
            result_paths = next_params["paths"]
        else:
            best_cost, best_parent = next_params
            break


    parent_cost, optimum_parent = best_cost, best_parent

    if optimum_parent not in result_paths:
        raise NoPathError

    parent_cost, optimum_parent, raw_optimum_blackboard = result_paths[optimum_parent]
    optimum_blackboard = dict(raw_optimum_blackboard)
    path = optimum_blackboard["src"]

    for parent_elem in path[-1::-1]:
        handle_backtrack_node(parent_elem)

    return best_cost, path
