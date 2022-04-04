from __future__ import annotations

import abc
import functools
import typing

from pyastar.mapconsts import WALL, OPEN, PATH, GOAL, CURR, SLOW
from pyastar.measures import manhattan_distance


def map_1():
    newmap = [
        [WALL, OPEN, WALL, WALL, WALL, WALL, WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN],
        [WALL, WALL, WALL, WALL, OPEN, WALL, WALL, OPEN],
        [OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN],
        [WALL, OPEN, OPEN, OPEN, OPEN, WALL, WALL],
        [WALL, OPEN, WALL, OPEN, OPEN, WALL, WALL, OPEN],
        [WALL, WALL, WALL, OPEN, WALL, WALL, WALL],
    ]
    return newmap


def map_2():
    newmap = [
        [WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, OPEN, OPEN],
        [WALL, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, WALL, OPEN],
        [WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, OPEN, WALL, OPEN],
        [WALL, OPEN, WALL, OPEN, OPEN, WALL, OPEN, OPEN, WALL, WALL],
        [WALL, OPEN, OPEN, OPEN, WALL, WALL, WALL, OPEN, OPEN, OPEN],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL],
        [OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL],
    ]
    return newmap


def map_3():
    newmap = [
        [WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, OPEN, OPEN, SLOW, OPEN, SLOW, OPEN, OPEN, WALL, OPEN, OPEN,
         OPEN, OPEN],
        [WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL, WALL],
        [WALL, OPEN, OPEN, OPEN, WALL, WALL, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, OPEN],
        [OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, OPEN],
        [WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN,
         OPEN, OPEN],
    ]
    return newmap


def map_4():
    newmap = [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN,
         WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL],
        [WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN,
         OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN,
         OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN,
         WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN,
         OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL],
        [WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN,
         OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL],
        [WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL,
         WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL,
         WALL],
        [WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, WALL, WALL, OPEN,
         WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, WALL, WALL, WALL, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN,
         WALL],
        [WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL,
         WALL, WALL, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN, WALL, WALL, WALL, OPEN,
         WALL],
        [WALL, OPEN, WALL, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN, WALL, OPEN, OPEN, OPEN, WALL, OPEN, WALL, OPEN, OPEN, OPEN,
         WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL,
         WALL]
    ]
    return newmap



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
    ) -> BasePathfindingGraph:

        return self


    @abc.abstractmethod
    def set_goal(
        self,
        pos,
        *args,
        **kwargs
    ) -> BasePathfindingGraph:

        return self


    @abc.abstractmethod
    def add_to_path(
        self,
        pos,
        *args,
        **kwargs
    ) -> BasePathfindingGraph:

        return self


    @abc.abstractmethod
    def visualize(self) -> BasePathfindingGraph:
        return self

    def is_passable(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        *args,
        **kwargs
    ):

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        passable = True

        return passable


class Map2D(BasePathfindingGraph):
    def __init__(self, raw_map=None, start_pos=None, diagonals=False):
        super().__init__(
            raw_map=raw_map or map_4(),
            start_pos=start_pos
        )
        self.diagonals = diagonals


    def __getitem__(self, item):
        x, y = item
        retval = self.map[y][x]
        return retval


    def __setitem__(self, key, value):
        x, y = key
        self.map[y][x] = value
        return


    def __str__(self) -> str:
        visualized_map = []

        for (y, row) in enumerate(self.map):
            out_row = []

            for (x, col) in enumerate(row):
                node = col

                if (x, y) == self.current_pos:
                    node = CURR

                elif (x, y) == self.current_goal:
                    node = GOAL

                elif (x, y) in self.path:
                    node = PATH

                out_row.append(node)

            visualized_map.append(out_row)

        return "\n".join("".join(row) for row in visualized_map)


    def adjacent_lazy(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        diagonals: bool | None = None,
        *args,
        **kwargs
    ) -> typing.Iterable[tuple[int, int]]:

        _diagonals = self.diagonals if diagonals is None else diagonals

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        x, y = _pos

        for y_pos in range(y-1, y+2):
            for x_pos in range(x-1, x+2):
                curr_pos = (x_pos, y_pos)

                if curr_pos not in self:
                    continue

                if curr_pos == pos:
                    continue

                if not _diagonals and abs(x_pos-x) == abs(y_pos-y):
                    continue

                if self.is_passable(curr_pos):
                    continue

                yield curr_pos
        return


    @functools.lru_cache(5)
    def adjacent(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        diagonals: bool | None = None,
        *args,
        **kwargs
    ) -> typing.Iterable[tuple[int, int]]:

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        positions = set(
            self.adjacent_lazy(
                pos=pos,
                pos_y=pos_y,
                diagonals=diagonals,
                *args,
                **kwargs
            )
        )

        return positions


    def set_current(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        *args,
        **kwargs
    ) -> Map2D:

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        self.current_pos = _pos
        return self


    def set_goal(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        *args,
        **kwargs
    ) -> Map2D:

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        self.current_goal = _pos
        return self


    def add_to_path(
        self,
        pos: float | tuple[float, float],
        pos_y: None | float = None,
        *args,
        **kwargs
    ) -> Map2D:

        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        self.path.append(_pos)
        return self


    def visualize(self, *args, **kwargs) -> Map2D:
        print("=" * 30)
        print(self)
        print("=" * 30)
        return self

    def is_passable(self, pos: float | tuple[float, float], pos_y: None | float = None, *args, **kwargs):
        if pos_y is not None:
            _pos = (pos, pos_y)
        else:
            _pos = pos

        passable = super().is_passable(_pos)

        if self[_pos] == WALL:
            passable = False

        return passable


class ActionGraph(BasePathfindingGraph):
    def __init__(self, raw_map=None, start_pos=None):
        super().__init__(
            raw_map=raw_map or reasoning_map(),
            start_pos=start_pos
        )


    def __getitem__(self, item):
        raw_path = item.split(",")
        path_iter = iter(raw_path)
        focus = self.map
        curr = NotImplemented
        while curr:
            curr = next(path_iter, None)
            if curr:
                focus = focus[curr]

        return focus


    def __setitem__(self, key, value):
        path = key.split(",")
        focus = self.map
        curr = NotImplemented
        while curr:
            curr = next(path, None)
            focus = focus[curr]
        else:
            focus[key] = value
        return


    def adjacent_lazy(self, pos, *args, **kwargs):
        focus = self.map
        adjacents = (k for k in focus.get(pos) or set())
        return adjacents


    def adjacent(self, pos, *args, **kwargs) -> typing.Iterable:
        adjacents = set(self.adjacent_lazy(pos, *args, **kwargs))
        return adjacents


    def set_goal(self, pos, *args, **kwargs) -> BasePathfindingGraph:
        self.current_goal = pos
        return self


    def set_current(self, pos, *args, **kwargs) -> BasePathfindingGraph:
        self.current_pos = pos
        return self


    def add_to_path(self, pos, *args, **kwargs) -> BasePathfindingGraph:
        self.path.append(pos)
        return self


    def visualize(self) -> BasePathfindingGraph:
        stored_path = self.path.copy()
        str_path = " => ".join((*map(str, stored_path), str(self.current_goal)))
        print(str_path)
        return self


def evaluate_neighbor(get_impassable, neigh, current_pos, goal, measure=None, neighbor_measure=None, goal_measure=None):
    _neighbor_measure = neighbor_measure or measure or manhattan_distance
    _goal_measure = goal_measure or measure or manhattan_distance

    impassability = get_impassable(neigh)

    if impassability:
        return float("inf")

    else:
        neigh_distance = _neighbor_measure(
            start=current_pos,
            end=neigh
        )

    goal_distance = _goal_measure(
        start=neigh,
        end=goal
    )

    heuristic = sum((
        neigh_distance,
        goal_distance
    ))

    # print(f"{current_pos}->{neigh} with cost {heuristic} ({neigh_distance}+{goal_distance})")

    return heuristic

