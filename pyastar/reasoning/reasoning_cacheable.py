import os
import typing

from pyastar.maputils import ActionGraph
from pyastar.impls.goap import cacheable_astar_solver
from pyastar.measures import no_goal_heuristic
from pyastar.reasoning.utils import State
from pyastar.reasoning.maps import load_map_json
from pyastar.types import ActionKey, ActionDict, StateLike

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

        act_preconds = preconds_fetcher(action) if isinstance(action, (str, ActionKey)) else action

        for state, value in act_preconds.items():
            if blackboard.get(state, 0) < value:
                match = False
                break

        return match

    return _checker


def main():
    start = State.fromdict({"HasDirtyDishes": 1}, name="START")
    # goal = State.fromdict({"Fed": 1}, name="END")
    # goal = State.fromdict({"Rested": 10}, name="END")
    # goal = State.fromdict({"Rested": 10, "Debug": 1}, name="END")
    # goal = State.fromdict({"Rested": 10}, name="END")
    # goal = State.fromdict({"Rested": 10, "Debug": 1}, name="END")
    goal = State.fromdict({"Debug": 1}, name="END")

    raw_map = custom_map()

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solve_astar = cacheable_astar_solver(adjacency_gen=get_actions(raw_map),
                                         preconditions_check=preconds_checker_for(raw_map),
                                         handle_backtrack_node=newmap.add_to_path,
                                         neighbor_measure=neighbor_measure(raw_map), goal_measure=no_goal_heuristic,
                                         goal_check=goal_checker_for(raw_map), get_effects=get_effects(raw_map),
                                         cutoff_iter=500, max_heap_size=100)

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
