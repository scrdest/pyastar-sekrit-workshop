import typing

from goapystar.impls.oop import BaseGOAP
from goapystar.measures import no_goal_heuristic
from goapystar.types import StateLike, IntoState, ActionKey
from goapystar.usecases.actiongraph.utils import get_actions, preconds_checker_for, neighbor_measure, goal_checker_for, \
    get_effects


class ActionGOAP(BaseGOAP):
    def __init__(self, mapobj: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapobj = mapobj
        self.path = []


    def adjacency_gen(self, *args, **kwargs) -> typing.Iterable:
        actiongetter = get_actions(self.mapobj)
        result = actiongetter()
        return result


    def preconditions_check(self, curr_state: StateLike, action: IntoState, *args, **kwargs) -> bool:
        precond_checker = preconds_checker_for(self.mapobj)
        result = precond_checker(curr_state, action)
        return result


    def handle_backtrack_node(self, action: ActionKey, *args, **kwargs):
        super().handle_backtrack_node(action, *args, **kwargs)
        self.path.append(action)


    def neighbor_measure(self, curr_state: StateLike, action: ActionKey, *args, **kwargs) -> float:
        neigh_measurer = neighbor_measure(self.mapobj)
        result = neigh_measurer(curr_state, action)
        return result


    def goal_measure(self, action: ActionKey, goal: StateLike, *args, **kwargs) -> float:
        goal_measurer = no_goal_heuristic
        result = goal_measurer(start=action, end=goal)
        return result


    def goal_check(self, curr_state: StateLike, goal: StateLike, *args, **kwargs) -> bool:
        goal_checker = goal_checker_for(self.mapobj)
        result = goal_checker(pos=curr_state, goal=goal)
        return result


    def get_effects(self, action: ActionKey, *args, **kwargs) -> StateLike:
        fx_getter = get_effects(self.mapobj)
        effects = fx_getter(action)
        return effects