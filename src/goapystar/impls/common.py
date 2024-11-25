import copy
import functools
import heapq
import operator
import typing

from ..measures import action_graph_dist, equality_check
from ..state import State
from ..types import StateLike, BlackboardBinOp, ActionKey, IntoState, PathTuple, CandidateTuple


class EmptyQueueError(Exception):
    pass


class NoPathError(Exception):
    pass


PLUS_INF = float("inf")
BLACKBOARD_CLASS = dict


def update_counts(
    src: StateLike,
    new: StateLike,
    default: typing.Any = 0,
    op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None
):
    base_op = op or operator.add
    op_is_dict = isinstance(op, dict)

    for new_key, new_val in new.items():
        op_for_key = op.get(new_key, base_op) if op_is_dict else base_op
        curr_value = src.get(new_key, default)
        src[new_key] = op_for_key(curr_value, new_val)

    return src


def evaluate_neighbor(
    check_preconds: typing.Callable[[IntoState, StateLike], bool],
    neigh: ActionKey,
    current_pos: IntoState,
    goal: StateLike,
    blackboard: typing.Optional[StateLike] = None,
    blackboard_default: typing.Any = 0,
    blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
    measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
    neighbor_measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
    goal_measure: typing.Optional[typing.Callable[[IntoState], float]] = None,
    get_effects: typing.Optional[typing.Callable[[ActionKey], StateLike]] = None,
):
    _neighbor_measure = neighbor_measure or measure or action_graph_dist
    _goal_measure = goal_measure or measure or action_graph_dist
    _blackboard = blackboard or BLACKBOARD_CLASS()

    valid = check_preconds(neigh, _blackboard)

    effects = copy.deepcopy(_blackboard or BLACKBOARD_CLASS())
    curr_src = list(effects.get("src") or [])
    curr_src.append(current_pos)

    if get_effects:
        new_effects = get_effects(neigh)
        update_counts(
            effects,
            new_effects,
            default=blackboard_default,
            op=blackboard_update_op,
        )

    effects["src"] = curr_src

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


def cached_parse_effects(effects_checker, blackboard_default=0, blackboard_update_op=None):

    @functools.lru_cache(10)
    def _check_effects(src_pos):
        rebuilt_blackboard = BLACKBOARD_CLASS()

        for trajectory in src_pos:
            traj_eff = effects_checker(trajectory)
            update_counts(
                rebuilt_blackboard,
                traj_eff,
                default=blackboard_default,
                op=blackboard_update_op
            )

        return rebuilt_blackboard

    return _check_effects


def _astar_deepening_search(
    start_pos: IntoState,
    goal: StateLike,
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionKey]],
    preconditions_checker: typing.Callable[[IntoState, StateLike], bool],
    max_queue_size: int = None,
    goal_checker: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    neighbor_measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
    goal_measure: typing.Optional[typing.Callable[[IntoState], float]] = None,
    pqueue_key_func: typing.Optional[typing.Callable[[int, float, float], tuple]] = None,
    blackboard: typing.Optional[StateLike] = None,
    blackboard_default: typing.Any = 0,
    blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
    visited: typing.Optional[typing.Dict[ActionKey, int]] = None,
    paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
    queue: typing.Optional[typing.MutableSequence[CandidateTuple]] = None,
    curr_cost: float = 0,
    _iter=1,
):

    _paths = paths or dict()
    _pqueue = queue or []
    _blackboard = blackboard.copy() if blackboard else BLACKBOARD_CLASS()
    _goal_check = goal_checker or equality_check

    if isinstance(start_pos, (State, dict)):
        update_counts(
            _blackboard,
            start_pos,
            default=blackboard_default,
            op=blackboard_update_op
        )

    if _goal_check(_blackboard, goal):
        cost, parent, path = _paths.get(start_pos) or (curr_cost, start_pos, (start_pos,))

        return False, (cost, _blackboard.get("src", []) + [start_pos])

    _neighbor_measure = neighbor_measure or action_graph_dist
    _goal_measure = goal_measure or action_graph_dist

    if visited is not None:
        visited[start_pos] = visited.get(start_pos, 0) + 1

    neighbors = adjacency_gen(start_pos)

    for neigh in neighbors:
        if visited and neigh in visited:
            continue

        heuristic, effects = evaluate_neighbor(
            check_preconds=preconditions_checker,
            neigh=neigh,
            current_pos=start_pos,
            goal=goal,
            neighbor_measure=_neighbor_measure,
            goal_measure=_goal_measure,
            blackboard=_blackboard,
            blackboard_default=blackboard_default,
            blackboard_update_op=blackboard_update_op,
            get_effects=get_effects
        )

        stored_neigh_cost, stored_curr_parent, _ = _paths.get(neigh) or (PLUS_INF, None, None)
        total_cost = curr_cost + heuristic
        src = effects["src"]

        if total_cost < stored_neigh_cost:
            _paths[neigh] = (total_cost, start_pos, src)

        # =================== VERY VERY *VERY* IMPORTANT: ===================
        # Storing the iteration as the first element of the candidate
        # tuple is *essential* for this to work properly.
        #
        # Why? Tuples compare with priority to the earlier items first.
        # By storing the iteration first, we enforce a BFS-like structure.
        #
        # Otherwise, cheap actions that don't meet the goal are expanded
        # before expensive actions that *do*; the algorithm logic is that
        # hopefully the cheap action will have a followup that satisfies
        # the search goal (in other words, depth-first search).
        # ===================================================================
        priority_key = pqueue_key_func(_iter, curr_cost, heuristic) if pqueue_key_func else (_iter,)
        cand_tuple = (priority_key, total_cost, neigh, src)

        if cand_tuple not in _pqueue and total_cost < PLUS_INF:
            heapq.heappush(
                _pqueue,
                cand_tuple
            )
            if max_queue_size is not None:
                _pqueue = _pqueue[:max_queue_size]

    if not _pqueue:
        raise EmptyQueueError("Exhausted all candidates before a path was found!")

    cand_cost, cand_pos, src_pos = heapq.heappop(_pqueue)[1:]

    fx_rebuilder = cached_parse_effects(
        get_effects,
        blackboard_default=blackboard_default,
        blackboard_update_op=blackboard_update_op
    )
    stack = tuple(src_pos + [cand_pos])
    cand_fx = fx_rebuilder(stack)

    cand_blackboard = cand_fx
    cand_blackboard["src"] = src_pos

    result = True, dict(
        start_pos=cand_pos,
        goal=goal,
        adjacency_gen=adjacency_gen,
        preconditions_checker=preconditions_checker,
        curr_cost=cand_cost,
        paths=_paths,
        visited=visited,
        neighbor_measure=_neighbor_measure,
        goal_measure=_goal_measure,
        queue=_pqueue,
        goal_checker=_goal_check,
        get_effects=get_effects,
        blackboard=cand_blackboard,
        blackboard_default=blackboard_default,
        blackboard_update_op=blackboard_update_op,
        max_queue_size=max_queue_size,
        _iter=_iter+1
    )
    return result


def suppress_not_found(default, default_factory=None):

    def _noexc_deco(func):

        def _safety_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)

            except NoPathError:
                result = default_factory() if default_factory else default

            return result

        return _safety_wrapper

    return _noexc_deco