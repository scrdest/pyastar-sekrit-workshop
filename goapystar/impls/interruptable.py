"""Goal Oriented Action Planning algorithm.

This is the interruptable variant.
The planner here is a lazy generator that yields current run's params.
This allows the search to be abandoned early, paused, or continued past the current point.
"""
import typing

from goapystar.impls.common import NoPathError, _astar_deepening_search, PLUS_INF
from goapystar.state import State
from goapystar.types import StateLike, ActionTuple, ActionKey, IntoState, PathTuple, BlackboardBinOp


def plan_interruptible(
    start_pos: IntoState,
    goal: IntoState,
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionTuple]],
    preconditions_check: typing.Callable[[StateLike], bool],
    handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
    paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
    visited: typing.Optional[typing.Dict[ActionKey, int]] = None,
    neighbor_measure: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    goal_measure: typing.Optional[typing.Callable[[IntoState], float]] = None,
    goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    cutoff_iter: typing.Optional[int] = 1000,
    max_queue_size: typing.Optional[int] = None,
    pqueue_key_func: typing.Optional[typing.Callable] = None,
    blackboard_default: typing.Any = 0,
    blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
):

    _start_pos = start_pos
    if not isinstance(start_pos, State):
        _start_pos = State.fromdict(start_pos, name="START")

    _goal = goal
    if not isinstance(goal, State):
        _goal = State.fromdict(goal, name="END")

    continue_search, next_params = True, dict(
        adjacency_gen=adjacency_gen,
        preconditions_checker=preconditions_check,
        start_pos=_start_pos,
        goal=_goal,
        paths=paths,
        visited=visited,
        neighbor_measure=neighbor_measure,
        goal_measure=goal_measure,
        goal_checker=goal_check,
        get_effects=get_effects,
        max_queue_size=max_queue_size,
        pqueue_key_func=pqueue_key_func,
        blackboard_default=blackboard_default,
        blackboard_update_op=blackboard_update_op,
    )

    best_cost, best_parent = None, None
    curr_iter = 0

    while next_params:
        continue_search, next_params = _astar_deepening_search(**next_params)

        if continue_search:

            if cutoff_iter is not None:
                curr_iter = next_params.get("_iter") or curr_iter + 1
                if curr_iter >= cutoff_iter:
                    raise NoPathError(f"Path not found within {cutoff_iter} iterations!")

                yield next_params

        else:
            best_cost, best_parent = next_params
            break

    parent_cost, path = best_cost, best_parent

    if handle_backtrack_node:
        for parent_elem in path:
            handle_backtrack_node(parent_elem)

    print(curr_iter)
    return best_cost, path


def find_plan(
    start_pos: IntoState,
    goal: IntoState,
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionTuple]],
    preconditions_check: typing.Callable[[StateLike], bool],
    handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
    paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
    visited: typing.Optional[typing.Dict[ActionKey, int]] = None,
    neighbor_measure: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    goal_measure: typing.Optional[typing.Callable[[IntoState], float]] = None,
    goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    cutoff_iter: typing.Optional[int] = 1000,
    max_queue_size: typing.Optional[int] = None,
    pqueue_key_func: typing.Optional[typing.Callable] = None,
    blackboard_default: typing.Any = 0,
    blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
):

    plan_loop = plan_interruptible(
        start_pos=start_pos,
        goal=goal,
        adjacency_gen=adjacency_gen,
        preconditions_check=preconditions_check,
        handle_backtrack_node=handle_backtrack_node,
        paths=paths,
        visited=visited,
        neighbor_measure=neighbor_measure,
        goal_measure=goal_measure,
        goal_check=goal_check,
        get_effects=get_effects,
        cutoff_iter=cutoff_iter,
        max_queue_size=max_queue_size,
        pqueue_key_func=pqueue_key_func,
        blackboard_default=blackboard_default,
        blackboard_update_op=blackboard_update_op
    )

    result = None
    params = None
    running = True

    while running:
        try:
            params = next(plan_loop)

        except StopIteration as stop:
            running = False
            result = stop.value

        except Exception:
            print(params)
            raise

    cost, path = result if result else (PLUS_INF, ())
    return cost, path


