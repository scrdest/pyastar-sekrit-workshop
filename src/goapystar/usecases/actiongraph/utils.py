import abc
import operator
import typing

from ...state import State
from ...types import ActionKey, ActionDict, StateLike


def get_actions(mapobj: ActionDict) -> typing.Callable:

    def _actiongetter(*args, **kwargs) -> typing.Sequence[ActionKey]:
        result = tuple(mapobj.keys())
        return result

    return _actiongetter


def get_effects(mapobj: ActionDict) -> typing.Callable:

    def _effectgetter(action, *args, **kwargs) -> typing.Sequence[State]:
        if isinstance(action, State):
            effects = action
        else:
            cost, preconds, effects = mapobj.get(action) or (float("inf"), State(), State())
        return effects

    return _effectgetter


def get_preconds(mapobj: ActionDict) -> typing.Callable[[ActionKey], typing.Sequence[State]]:

    def _actiongetter(action: ActionKey, *args, **kwargs) -> typing.Sequence[State]:
        cost, preconds, effects = mapobj[action]
        return preconds

    return _actiongetter


def neighbor_measure(mapobj: ActionDict) -> typing.Callable:

    def _measurer(start, end: ActionKey) -> float:
        cost, preconds, effects = mapobj[end]
        return cost

    return _measurer


def goal_checker_for(mapobj: ActionDict, cmp_op=operator.gt) -> typing.Callable:
    effect_getter = get_effects(mapobj=mapobj)

    def _goalchecker(pos: StateLike, goal: StateLike) -> bool:
        match = True
        pos_effects = effect_getter(pos) if isinstance(pos, str) else pos

        for state, value in goal.items():
            if value is None:
                continue

            if cmp_op(value, pos_effects.get(state, 0)):
                match = False
                break

            match = True

        return match

    return _goalchecker


def preconds_checker_for(mapobj: ActionDict) -> typing.Callable[[typing.Union[ActionKey, StateLike], StateLike], bool]:
    preconds_fetcher = get_preconds(mapobj=mapobj)

    def _checker(action, blackboard):
        match = True

        act_preconds = preconds_fetcher(action) if isinstance(action, (str, ActionKey)) else action

        for state, value in act_preconds.items():
            if blackboard.get(state, 0) < value:
                match = False
                break

        return match

    return _checker


def reasoning_map():
    Nodename = str
    Distance = typing.Union[int, float]

    action_map: dict[Nodename, dict[Nodename, Distance]] = {
        "1": {"2": 1, "3": 3},
        "2": {"3": 1, "1": 1},
        "3": {"4": 2, "5": 2, "6": 20},
        "4": {"2": 1, "5": 4},
        "5": {"6": 20, "1": 3},
    }
    return action_map


class BasePathfindingGraph:
    def __init__(self, raw_map=None, start_pos=None, *args, **kwargs):
        self.map = raw_map
        self.current_pos = start_pos
        self.current_goal = None
        self.path = list()


    def __contains__(self, item: tuple):
        lens = self.map
        for dim in item[::-1]:
            if dim < 0:
                return False

            if dim >= len(lens):
                return False

            lens = lens[dim]
        return True


    def __iter__(self):
        return iter(self.map)


    @abc.abstractmethod
    def __getitem__(self, item):
        return


    @abc.abstractmethod
    def __setitem__(self, key, value):
        return

    @abc.abstractmethod
    def adjacent_lazy(
        self,
        pos,
        *args,
        **kwargs
    ):
        yield None


    @abc.abstractmethod
    def adjacent(
        self,
        pos,
        *args,
        **kwargs
    ) -> typing.Iterable:

        return set(self.adjacent_lazy(pos=pos, *args, **kwargs))


    @abc.abstractmethod
    def set_current(
        self,
        pos,
        *args,
        **kwargs
    ) -> 'BasePathfindingGraph':

        return self


    @abc.abstractmethod
    def set_goal(
        self,
        pos,
        *args,
        **kwargs
    ) -> 'BasePathfindingGraph':

        return self


    @abc.abstractmethod
    def add_to_path(
        self,
        pos,
        *args,
        **kwargs
    ) -> 'BasePathfindingGraph':

        return self


    @abc.abstractmethod
    def visualize(self) -> 'BasePathfindingGraph':
        return self


    def is_passable(
        self,
        pos: typing.Union[float, typing.Tuple[float, float]],
        pos_y: typing.Optional[float] = None,
        *args,
        **kwargs
    ):

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        passable = True

        return passable


    def is_impassable(
        self,
        pos: typing.Union[float, typing.Tuple[float, float]],
        pos_y: typing.Optional[float] = None,
        *args,
        **kwargs
    ):
        passable = not self.is_passable(
            pos=pos,
            pos_y=pos_y,
            *args,
            **kwargs
        )
        return passable
