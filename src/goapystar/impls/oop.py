"""Goal Oriented Action Planning algorithm.

This is the base class for the OOP API.
For actual premade implementations, see the goapystar.usecases package.
"""

import abc
import functools
import typing

from .common import NoPathError, PLUS_INF, _astar_deepening_search
from ..state import State
from ..types import StateLike, ActionTuple, IntoState, BlackboardBinOp, ActionKey, PathTuple, ResultTuple


class BaseGOAP(abc.ABC):
    cutoff_iter = 20000
    max_queue_size = None
    blackboard_default = 0

    def __init__(
        self,
        adjacency_gen: typing.Optional[typing.Callable[[StateLike], typing.Iterable[ActionTuple]]] = None,
        preconditions_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
        handle_backtrack_node: typing.Optional[typing.Callable[[ActionTuple], typing.Any]] = None,
        neighbor_measure: typing.Optional[typing.Callable[[StateLike], bool]] = None,
        goal_measure: typing.Optional[typing.Callable[[IntoState], float]] = None,
        goal_check: typing.Optional[typing.Callable[[StateLike], bool]] = None,
        get_effects: typing.Optional[typing.Callable[[StateLike], float]] = None,
        cutoff_iter: typing.Optional[int] = None,
        max_queue_size: typing.Optional[int] = None,
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
        self.max_queue_size = max_queue_size or self.max_queue_size
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


    def find_plan(
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
            max_queue_size=self.max_queue_size,
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
                self.handle_backtrack_node(parent_elem)

        return best_cost, path


    def __call__(self, *args, **kwargs) -> ResultTuple:
        return self.find_plan(*args, **kwargs)
