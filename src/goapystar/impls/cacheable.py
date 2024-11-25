"""Goal Oriented Action Planning algorithm.

This is the cacheable variant.
The planner here is a closure with all the non-cacheable params (callbacks, mostly)
This means we can apply lru_cache() to the function returned by the cacheable_solver()
     and reap the benefits of in-memory caching of paths.
Realistically, this *probably* isn't that useful, since states are path-dependent, but it is a PoC.
"""
import functools
import typing

from .common import NoPathError, _astar_deepening_search
from ..state import State
from ..types import StateLike, ActionTuple, IntoState, BlackboardBinOp, ActionKey, PathTuple, ResultTuple


def cacheable_solver(
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionTuple]],
    preconditions_check: typing.Callable[[StateLike], bool],
    handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
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

    def cacheable_solve(
        start_pos: IntoState,
        goal: StateLike,
        paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
    ) -> ResultTuple:

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

            else:
                best_cost, best_parent = next_params
                break

        parent_cost, path = best_cost, best_parent

        for parent_elem in path:
            handle_backtrack_node(parent_elem)

        return best_cost, path

    return cacheable_solve


def cached_solver(cache_size=None, *args, **kwargs):
    uncached_solver = cacheable_solver(*args, **kwargs)
    _cached_solver = functools.lru_cache(maxsize=cache_size)(uncached_solver)
    return _cached_solver


def find_plan(cache_size=None, setup_args=None, setup_kwargs=None, *args, **kwargs):
    _setup_args = setup_args or []
    _setup_kwargs = setup_kwargs or {}

    solver = cached_solver(cache_size=cache_size, *setup_args, **setup_kwargs)

    cost, path = solver(*args, **kwargs)

    return cost, path
