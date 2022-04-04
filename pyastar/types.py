import typing

from pyastar.reasoning.utils import State

StateLike = typing.Union[dict, State]

Cost = float
Preconditions = StateLike
Effects = StateLike

ActionTuple = typing.Tuple[Cost, Preconditions, Effects]
IntoState = typing.Union[StateLike, ActionTuple]
