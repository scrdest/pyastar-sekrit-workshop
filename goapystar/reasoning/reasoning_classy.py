import os
import typing

from goapystar.maputils import BasePathfindingGraph, ActionGraph
from goapystar.impls.goap import BaseGOAP
from goapystar.reasoning.utils import State
from goapystar.reasoning.maps import load_map_json
from goapystar.measures import no_goal_heuristic
from goapystar.types import ActionKey, ActionDict, ActionTuple, StateLike, IntoState

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
MAPDIR_SUFF = "maps"
MAPS_DIR = os.path.join(CURR_DIR, MAPDIR_SUFF)


def custom_map() -> ActionDict:
    actionmap = load_map_json("map1")
    return actionmap


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


def goal_checker_for(mapobj: ActionDict) -> typing.Callable:
    effect_getter = get_effects(mapobj=mapobj)

    def _goalchecker(pos: StateLike, goal: StateLike) -> bool:
        match = True
        pos_effects = effect_getter(pos) if isinstance(pos, str) else pos

        for state, value in goal.items():
            if value is None:
                continue

            if pos_effects.get(state, 0) < value:
                match = False
                break

            match = True

        return match

    return _goalchecker


def preconds_checker_for(mapobj: ActionDict) -> typing.Callable[[typing.Union[ActionKey, StateLike], StateLike], bool]:
    preconds_fetcher = get_preconds(mapobj=mapobj)

    def _checker(action, blackboard):
        match = True

        act_preconds = preconds_fetcher(action) if isinstance(action, str) else action

        for state, value in act_preconds.items():
            if blackboard.get(state, 0) < value:
                match = False
                break

        return match

    return _checker



class ReasoningGOAP(BaseGOAP):
    def __init__(self, mapobj: BasePathfindingGraph, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapobj = mapobj


    def adjacency_gen(self, *args, **kwargs) -> typing.Iterable:
        actiongetter = get_actions(self.mapobj.map)
        result = actiongetter()
        return result


    def preconditions_check(self, curr_state: StateLike, action: IntoState, *args, **kwargs) -> bool:
        precond_checker = preconds_checker_for(self.mapobj.map)
        result = precond_checker(curr_state, action)
        return result


    def handle_backtrack_node(self, action: ActionTuple, *args, **kwargs):
        super().handle_backtrack_node(action, *args, **kwargs)
        self.mapobj.add_to_path(action)


    def neighbor_measure(self, curr_state: StateLike, action: ActionTuple, *args, **kwargs) -> float:
        neigh_measurer = neighbor_measure(self.mapobj.map)
        result = neigh_measurer(curr_state, action)
        return result


    def goal_measure(self, action: ActionTuple, goal: StateLike, *args, **kwargs) -> float:
        goal_measurer = no_goal_heuristic
        result = goal_measurer(start=action, end=goal)
        return result


    def goal_check(self, curr_state: StateLike, goal: StateLike, *args, **kwargs) -> bool:
        goal_checker = goal_checker_for(self.mapobj.map)
        result = goal_checker(pos=curr_state, goal=goal)
        return result


    def get_effects(self, action: ActionTuple, *args, **kwargs) -> StateLike:
        fx_getter = get_effects(self.mapobj.map)
        effects = fx_getter(action)
        return effects


def main():
    start = {"HasDirtyDishes": 1}
    goal = {"Debug": 1}

    raw_map = custom_map()

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solver = ReasoningGOAP(
        mapobj=newmap,
        cutoff_iter=500,
        max_heap_size=100,
    )

    cost, path = solver(
        start_pos=start,
        goal=goal,
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
