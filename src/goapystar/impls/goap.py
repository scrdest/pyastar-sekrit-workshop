"""Goal Oriented Action Planning algorithm.

For the uninitiated: GOAP is a STRIPS-like, search-based planning algorithm
typically used for game AIs, originally implemented by Jeff Orkin for the 2005
videogame F.E.A.R. and subsequently featured in several other AAA releases.

Unlike the more traditional approaches that rely on the programmer to implement
sane transitions between actions available to the agent, GOAP takes a sequence of
actions (with preconditions for their execution and postconditions for their effects
defined) available to the agent and finds its own way into a solution using the
classic A* algorithm over the graph of actions.

The result is an almost *notoriously* 'smart' and somewhat unpredictable AI agent;
for instance, STALKER's initial implementation had to be *artificially limited* to
keep NPCs from *winning the game ahead of the player*.

This implementation is somewhat more sophisticated form of the algorithm, closer to
classic STRIPS - the goals and conditions are specified as numeric values, which means
the algorithm can account for fuzzy resources, e.g. funds or health, rather than
binary states, without resorting to creating boilerplate boolean flags for each
variant (e.g. HasMoney1, HasMoney20, HasMoney100) which is inflexible and inelegant.



# On the core Astar implementation:

This is a fancy recursive-ish implementation; instead of carrying a
huge stack around, we use a pure-ish function and pass *parameters*
around so that the outputs of one call serve as inputs to the next.

The core 'worker' function is wrapped in a 'manager' function that handles
scheduling, ferries parameters between calls, and handles the backtracking phase.

This is approximately Trampolining (in the Lisp sense) - we just return the parameters
to the function we need called rather than a proper thunk, because data is nicer to debug
and serde rather than dealing with raw continuations. It's not like we ever need to call
any *other* functions than the next worker call anyway.
"""
import typing

from.common import NoPathError, PLUS_INF, _astar_deepening_search, suppress_not_found
from ..state import State
from ..types import StateLike, ActionTuple, ActionKey, IntoState, PathTuple, BlackboardBinOp


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

        else:
            best_cost, best_parent = next_params
            break

    parent_cost, path = best_cost, best_parent

    if handle_backtrack_node:
        for parent_elem in path:
            handle_backtrack_node(parent_elem)

    return best_cost, path


@suppress_not_found(default=None, default_factory=lambda: (PLUS_INF, list()))
def maybe_find_plan(*args, **kwargs):
    return find_plan(*args, **kwargs)


