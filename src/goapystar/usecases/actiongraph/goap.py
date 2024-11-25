from ..actions import ActionGOAP
from .utils import BasePathfindingGraph
from ...types import ActionKey


class FancyActionGraphGOAP(ActionGOAP):
    def __init__(self, mapobj: BasePathfindingGraph, *args, **kwargs):
        super().__init__({}, *args, **kwargs)
        self.mapobj = mapobj.map
        self.graph = mapobj

    def handle_backtrack_node(self, action: ActionKey, *args, **kwargs):
        super().handle_backtrack_node(action, *args, **kwargs)
        self.graph.add_to_path(action)
