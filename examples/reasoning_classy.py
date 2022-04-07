import typing

from goapystar.map_2d.utils import BasePathfindingGraph, ActionGraph
from goapystar.maputils import load_map_json
from goapystar.measures import no_goal_heuristic
from goapystar.impls.goap import BaseGOAP
from goapystar.types import ActionDict, ActionTuple, StateLike, IntoState
from goapystar.actiongraph.utils import (
    get_actions,
    get_effects,
    neighbor_measure,
    preconds_checker_for,
    goal_checker_for
)


def custom_map() -> ActionDict:
    actionmap = load_map_json("map1")
    return actionmap


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
    goal = {"Debug": 1, "HasFood": 1}

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
