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
import abc
import copy
import functools
import heapq
import operator
import typing

from goapystar.reasoning.utils import State
from goapystar.measures import action_graph_dist, equality_check
from goapystar.types import StateLike, ActionTuple, ActionKey, IntoState, CandidateTuple, PathTuple, ResultTuple, \
    BlackboardBinOp


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
    goal_measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
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
    max_heap_size: int = None,
    goal_checker: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    neighbor_measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
    goal_measure: typing.Optional[typing.Callable[[StateLike], float]] = None,
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
            if max_heap_size is not None:
                _pqueue = _pqueue[:max_heap_size]

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
        max_heap_size=max_heap_size,
        _iter=_iter+1
    )
    return result


def solve_astar(
    start_pos: IntoState,
    goal: IntoState,
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionTuple]],
    preconditions_check: typing.Callable[[StateLike], bool],
    handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
    paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
    visited: typing.Optional[typing.Dict[ActionKey, int]] = None,
    neighbor_measure: typing.Callable[[StateLike], bool] = None,
    goal_measure: typing.Callable[[StateLike], bool] = None,
    goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    cutoff_iter: typing.Optional[int] = 1000,
    max_heap_size: typing.Optional[int] = None,
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
        max_heap_size=max_heap_size,
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

    print(curr_iter)
    return best_cost, path


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


@suppress_not_found(default=None, default_factory=lambda: (PLUS_INF, list()))
def maybe_solve_astar(*args, **kwargs):
    return solve_astar(*args, **kwargs)


def cacheable_astar_solver(
    adjacency_gen: typing.Callable[[StateLike], typing.Iterable[ActionTuple]],
    preconditions_check: typing.Callable[[StateLike], bool],
    handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
    neighbor_measure: typing.Callable[[StateLike], bool] = None,
    goal_measure: typing.Callable[[StateLike], bool] = None,
    goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
    get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
    cutoff_iter: typing.Optional[int] = 1000,
    max_heap_size: typing.Optional[int] = None,
    pqueue_key_func: typing.Optional[typing.Callable] = None,
    blackboard_default: typing.Any = 0,
    blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
):

    def cacheable_solve_astar(
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
            max_heap_size=max_heap_size,
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

    return cacheable_solve_astar


class BaseGOAP(abc.ABC):
    cutoff_iter = 1000
    max_heap_size = None
    blackboard_default = 0

    def __init__(
        self,
        adjacency_gen: typing.Optional[typing.Callable[[StateLike], typing.Iterable[ActionTuple]]] = None,
        preconditions_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
        handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
        neighbor_measure: typing.Callable[[StateLike], bool] = None,
        goal_measure: typing.Callable[[StateLike], bool] = None,
        goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
        get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
        cutoff_iter: typing.Optional[int] = None,
        max_heap_size: typing.Optional[int] = None,
        pqueue_key_func: typing.Optional[typing.Callable] = None,
        blackboard_default: typing.Any = None,
        blackboard_update_op: typing.Optional[typing.Union[BlackboardBinOp, typing.Dict[ActionKey, BlackboardBinOp]]] = None,
        *args,
        **kwargs,
    ):
        # All configuration options can be overridden at init if so desired.
        self.adjacency_gen = adjacency_gen or self.adjacency_gen
        self.preconditions_check = preconditions_check or self.preconditions_check
        self.handle_backtrack_node = handle_backtrack_node or self.handle_backtrack_node
        self.neighbor_measure = neighbor_measure or self.neighbor_measure
        self.goal_measure = goal_measure or self.goal_measure
        self.goal_check = goal_check or self.goal_check
        self.get_effects = get_effects or self.get_effects
        self.cutoff_iter = cutoff_iter or self.cutoff_iter
        self.max_heap_size = max_heap_size or self.max_heap_size
        self.blackboard_default = blackboard_default or None
        self.blackboard_update_op = blackboard_update_op or None
        self.pqueue_key_func = pqueue_key_func or None


    @abc.abstractmethod
    def adjacency_gen(self, *args, **kwargs) -> typing.Iterable[ActionTuple]:
        return (i for i in [])


    @abc.abstractmethod
    def preconditions_check(self, curr_state: StateLike, action: ActionTuple, *args, **kwargs) -> bool:
        return False


    @abc.abstractmethod
    def handle_backtrack_node(self, action: ActionTuple, *args, **kwargs):
        return


    @abc.abstractmethod
    def neighbor_measure(self, curr_state: StateLike, action: ActionTuple, *args, **kwargs) -> float:
        return PLUS_INF


    @abc.abstractmethod
    def goal_measure(self, action: ActionTuple, goal: StateLike, *args, **kwargs) -> float:
        return PLUS_INF


    @abc.abstractmethod
    def goal_check(self, curr_state: StateLike, goal: StateLike, *args, **kwargs) -> bool:
        return False


    @abc.abstractmethod
    def get_effects(self, action: ActionTuple, *args, **kwargs) -> StateLike:
        return {}


    def partial_with_self(self, func):
        @functools.wraps(func)
        def _selfie_wrapper(*args, **kwargs):
            return func(self, *args, **kwargs)
        return _selfie_wrapper


    def plan(
        self,
        start_pos: IntoState,
        goal: StateLike,
        paths: typing.Optional[typing.Dict[ActionKey, PathTuple]] = None,
        *args,
        **kwargs
    ) -> ResultTuple:

        _start_pos = start_pos
        if not isinstance(start_pos, State):
            _start_pos = State.fromdict(start_pos, name="START")

        _goal = goal
        if not isinstance(goal, State):
            _goal = State.fromdict(goal, name="END")


        continue_search, next_params = True, dict(
            adjacency_gen=self.adjacency_gen,
            preconditions_checker=self.preconditions_check,
            start_pos=_start_pos,
            goal=_goal,
            paths=paths,
            neighbor_measure=self.neighbor_measure,
            goal_measure=self.goal_measure,
            goal_checker=self.goal_check,
            get_effects=self.get_effects,
            max_heap_size=self.max_heap_size,
            pqueue_key_func=self.pqueue_key_func,
        )

        best_cost, best_parent = None, None
        curr_iter = 0

        while next_params:
            continue_search, next_params = _astar_deepening_search(**next_params)

            if continue_search:

                if self.cutoff_iter is not None:
                    curr_iter = next_params.get("_iter") or curr_iter + 1
                    if curr_iter >= self.cutoff_iter:
                        raise NoPathError(f"Path not found within {self.cutoff_iter} iterations!")

            else:
                best_cost, best_parent = next_params
                break

        parent_cost, path = best_cost, best_parent

        if self.handle_backtrack_node:
            for parent_elem in path:
                self.handle_backtrack_node(self, parent_elem)

        return best_cost, path


    def __call__(self, *args, **kwargs) -> ResultTuple:
        return self.plan(*args, **kwargs)

