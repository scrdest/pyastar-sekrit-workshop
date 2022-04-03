import types
import typing

from pyastar.maputils import ActionGraph
from pyastar.impls.goap_v2 import solve_astar

def custom_map():
    actionmap = {
        "Eat": (1, State.fromdict({"HasFood": 1, "HasCleanDishes": 1}, "eat"), State.fromdict({"HasDirtyDishes": 1, "HasCleanDishes": -1, "Fed": 1, "HasFood": -1})),
        "Shop": (1, State.fromdict({"Money": 10}), State.fromdict({"HasFood": 1, "Money": -10})),
        "Party": (1, State.fromdict({"Money": 11}), State.fromdict({"Rested": 1, "Money": -11})),
        "Sleep": (1, State.fromdict({"Fed": 1, }), State.fromdict({"Rested": 10})),
        "DishWash": (1, State.fromdict({"HasDirtyDishes": 1, "Rested": 1}), State.fromdict({"HasDirtyDishes": -1, "HasCleanDishes": 1, "Rested": -1})),
        "Work": (1, State.fromdict({"Rested": 1}), State.fromdict({"Money": 10})),
        "Idle": (1, State(name="idle"), State.fromdict({"Rested": 1})),
        "DebugGetSimple": (100, State(), State(Debug=1)),
        # "DebugCheapFeed": (1, State(), State(Fed=1)),
        # "DebugExpensiveFeed": (100, State(), State(Fed=1)),
    }

    return actionmap


def check_preconditions_for_graph(mapobj: dict):

    def _checker(neighbor, blackboard):

        result = True
        cost, preconds, effects = mapobj[neighbor]

        for prec_key, prec_value in preconds.items():
            if blackboard.get(prec_key, 0) < prec_value:
                return False

        return result

    return _checker


def get_actions(mapobj):

    def _actiongetter(*args, **kwargs):
        result = tuple(mapobj.keys())
        return result

    return _actiongetter


def get_effects(mapobj):

    def _actiongetter(action, *args, **kwargs):
        cost, preconds, effects = mapobj.get(action) or (float("inf"), State(), State())
        return effects

    return _actiongetter


def get_preconds(mapobj):

    def _actiongetter(action, *args, **kwargs):
        cost, preconds, effects = mapobj[action]
        return preconds

    return _actiongetter


def neighbor_measure(mapobj):

    def _measurer(start, end):
        cost, preconds, effects = mapobj[end]
        return cost

    return _measurer


class State(types.SimpleNamespace):
    def __init__(self, name=None, **values):
        super().__init__()
        self.__dict__.update(values)
        self._name = name or "State"

    @classmethod
    def fromdict(cls, initdict: dict, name=None):
        inst = cls()
        inst.__dict__.update(initdict)
        if name:
            inst._name = name
        return inst

    def keys(self):
        return (k for k in self.__dict__.keys() if not k[0][0] == "_")

    def items(self):
        return (i for i in self.__dict__.items() if not i[0][0] == "_")

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __getattr__(self, item):
        return self.__dict__[item]

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return other.__dict__ == self.__dict__

        return super().__eq__(other)

    def __hash__(self):
        return id(self)

    def __str__(self):
        stringform = f"<{self._name} ({dict(self.items())})>"
        return stringform


def goal_checker_for(mapobj):
    effect_getter = get_effects(mapobj=mapobj)

    def _goalchecker(pos, goal):
        match = False
        pos_effects = effect_getter(pos) if isinstance(pos, str) else pos

        for state, value in goal.items():
            if pos_effects.get(state, 0) < value:
                match = False
                break
            match = True

        return match

    return _goalchecker


def preconds_checker_for(mapobj) -> typing.Callable[[typing.Union[str, dict, State], dict], bool]:
    preconds_checker = get_preconds(mapobj=mapobj)

    def _checker(action, blackboard):
        match = True

        act_preconds = preconds_checker(action) if isinstance(action, str) else action

        for state, value in act_preconds.items():
            if blackboard.get(state, 0) < value:
                match = False
                break

        return match

    return _checker


def main():
    start = "START"
    # goal = State.fromdict({"Fed": 1}, name="fed")
    # goal = State.fromdict({"Rested": 10}, name="rested")
    # goal = State.fromdict({"Rested": 10, "Debug": 1}, name="rested")
    goal = State.fromdict({"Rested": 10}, name="rested")
    # goal = State.fromdict({"Debug": 1}, name="fed")

    # raw_map = reasoning_map()
    raw_map = custom_map()

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    cost, path = solve_astar(
        start_pos=start,
        goal=goal,
        adjacency_gen=get_actions(raw_map),
        preconditions_check=preconds_checker_for(raw_map),
        handle_backtrack_node=lambda parent: newmap.add_to_path(parent),
        neighbor_measure=neighbor_measure(raw_map),
        goal_measure=lambda start, end: 1,
        goal_check=goal_checker_for(raw_map),
        get_effects=get_effects(raw_map),
        blackboard={"HasDirtyDishes": 1}
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
