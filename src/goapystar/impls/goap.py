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
    use_transposition_table: bool = True,
):
    """Run a GOAP planner to achieve a specified goal state given an initial state.
    This is an NP-hard problem; the planner is NOT guaranteed to find a plan in a sane amount of time.
    As such, you can specify a planning budget; the planner can either return a (cost, plan) pair
    or throw a NoPathError exception if no plan was found within the assigned budget.

    :param start_pos: Initial state, either a State class from this package or just a plain old dict.
                      In the latter case, keys are ideally strings, values ideally float or int.
    :param goal: The state we want to have after the final action in a valid plan.
                 Depending on goal_check, may only be a minimum (i.e. can overperform).
                 Either a State class from this package or just a plain old dict.
                 In the latter case, keys are ideally strings, values ideally float or int.
    :param adjacency_gen: A callable that, given an action key, returns an iterable
                          of 'neighboring' action keys we can reach from there.
                          In other words, returns all Actions we could possibly take after
                          we finish doing the current Action. (you may, but needn't,
                          account for the neighbors' preconditions being satisfied)
    :param preconditions_check: A callable that takes an Action key and a forecasted state and returns
                                a boolean indicating if the preconditions of that Action are met.
                                E.g. if an OpenDoor Action requires a Key,
                                this function should implement checking if we have a Key.
    :param handle_backtrack_node: Optional. Callback to apply to each node in the found Plan as we report back.
                                  Mainly used for diagnostics, use only if you know exactly what you're doing.
    :param paths: Optional. A dictionary of current partial candidate Plans.
                  Mainly used by the guts of the code, but shouldn't break if
                  you have a good use-case for supplying a custom value here.
    :param visited: Optional. A dictionary of already visited nodes to num. visits.
                    Mainly used by the guts of the code, but shouldn't break if
                    you have a good use-case for supplying a custom value here.
    :param neighbor_measure: Optional. A callable that takes a current and neighbor Actions and returns
                             a heuristic for how promising that neighbor is as the next move to explore.
                             By default uses an unbiased heuristic that treats most moves equally.
                             A custom heuristic tailored to a specific application can speed planning
                             up massively though!
    :param goal_measure: Optional. A callable that takes a neighbor Actions and *the goal state* and returns
                         a heuristic for how promising that neighbor is as the next move to explore.
                         By default uses an unbiased heuristic that treats most moves equally.
                         A custom heuristic tailored to a specific application can speed planning
                         up massively though!
                         *This is the distance to the goal, while neighbor_measure is distance to where we are NOW.*
    :param goal_check: Optional. A callable that takes in the projected state and the goal state and returns
                       True if the projected situation satisfies our goal, False otherwise.
                       This can be used to choose whether you want:
                       - an *exact* solution (default, equality check),
                       - at least as good maximizer solution (allows overshooting the goal, greater-or-equal check)
                       - at least as good minimizer solution (as above, but keeps the goal within a 'budget'; less-than)
                       - anything else, e.g. a mix of all the above depending on which key we're looking at.
    :param get_effects: Optional. A callable that takes an Action and returns its Effects.
    :param cutoff_iter: Optional. Budget for number of planning iterations. Default 1000.
                        This is the main dial controlling how much resources the planner can consume.
                        Lower values guarantee the planner won't spin forevermore, but more likely to fail.
                        Higher conversely are more likely to find a plan if there is a valid solution, but
                        if there isn't, you're just wasting more of your time and CPU cycles in vain.
                        Sadly, the right value to use depends heavily on *all the other parameters*.
                        The test cases all generally finish within 20k iterations at worst.
    :param max_queue_size: Optional. Memory budget. If set (off by default), the search algorithm becomes a beam search.
                           This will limit the maximum RAM consumption.
                           The effect on the results is a gamble - sometimes you may get a solution faster,
                           sometimes it will fail to find a solution where it succeeded before.
                           Broadly, more complex, multi-step plans require a higher budget (or no limit).
    :param pqueue_key_func: Optional. A callable that takes in a planning iteration, cost, and heuristic
                            and returns the Priority for the queue of candidates.
                            By default - just uses the iteration (breadth-first search).
    :param blackboard_default: Optional. What value to use for the state keys by default.
                               Must be of a compatible type with the state values and the update op
                               (e.g. if you decide to use string values, this should be a string too).
    :param blackboard_update_op: Optional. Specifies how to update the state with the Effects of Actions.
                                 By default, assumes the state values are numbers and effects are additive, so
                                 an Action with effect of 'Food: +3' will increase the current value of Food state by 3.
                                 Alternatives include overriding (we'd set Food value TO 3 no matter what it was before)
                                 for stateless effects, add-negative when effects are costs (and you're too lazy to type
                                 minus all the time), operators for custom data types, or whatever else.
                                 This parameter also accepts a dictionary of {key: func} if you want different state
                                 keys to have different merging logic.
    :param use_transposition_table: Optional boolean. If True (default) uses an optimization that discards duplicate
                                    states - if we have two equivalent ways to get to the same place, ignore the second.
                                    This generally makes planning absurdly more efficient for the same budget.
                                    Some reasons you might want to disable this are:
                                    1) Your use-case DOES care about paths that are different but equivalent, somehow.
                                    2) You have really large states and discover the hashing required is a bottleneck.
                                    3) You need a coffee break so you want the code to run slower.
    :raises: A NoPathError if no solution was found within the budget
    :return: A (cost, plan) tuple if a plan was found.
    """

    _start_pos = start_pos
    if not isinstance(start_pos, State):
        _start_pos = State.fromdict(start_pos, name="START")

    _goal = goal
    if not isinstance(goal, State):
        _goal = State.fromdict(goal, name="END")

    transposition_table = None

    if use_transposition_table:
        transposition_table = {_start_pos.as_hash()}

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
        transposition_table=transposition_table,
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

