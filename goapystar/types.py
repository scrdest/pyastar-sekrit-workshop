import typing

from goapystar.reasoning.utils import State

StateLike = typing.Union[dict, State]

Cost = float
Preconditions = StateLike
Effects = StateLike

ActionKey = typing.NewType("ActionKey", str)
ActionTuple = typing.Tuple[Cost, Preconditions, Effects]
ActionItem = typing.Tuple[ActionKey, ActionTuple]
ActionDict = typing.Dict[ActionKey, ActionTuple]

IntoState = typing.Union[StateLike, ActionTuple, ActionKey]

PathTuple = typing.Tuple[Cost, ActionKey, typing.Sequence[ActionKey]]
CandidateTuple = typing.Tuple[typing.Any, Cost, ActionKey, typing.Sequence[ActionKey]]
ResultTuple = typing.Tuple[Cost, typing.Sequence[IntoState]]

BlackboardBinOp = typing.Callable[[dict, dict], typing.Any]