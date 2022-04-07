import operator
import typing

from goapystar.state import State
from goapystar.types import ActionKey, ActionDict, StateLike


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

