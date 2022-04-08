import typing

from goapystar.usecases.actiongraph.utils import reasoning_map, BasePathfindingGraph


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
        if self.current_goal:
            stored_path.append(self.current_goal)

        str_path = " => ".join(map(str, stored_path))
        print(str_path)
        return self
