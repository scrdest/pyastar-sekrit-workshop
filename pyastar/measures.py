import typing

from pyastar.mapconsts import SLOW


def manhattan_distance(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:
    dist = sum((
        abs(end_dim - start_dim)
        for (end_dim, start_dim)
        in zip(end, start)
    ))
    return dist


def euclidean_distance(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:
    dist = sum((
        (end_dim - start_dim) ** 2
        for (end_dim, start_dim)
        in zip(end, start)
    )) ** 0.5
    return dist


def chebyshev_distance(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:
    dist = max((
        abs(end_dim - start_dim)
        for (end_dim, start_dim)
        in zip(end, start)
    ))
    return dist


def antichebyshev_distance(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:
    dist = min((
        abs(end_dim - start_dim)
        for (end_dim, start_dim)
        in zip(end, start)
    ))
    return dist


def minkowski_distance(
    dims: float
) -> typing.Callable[[typing.Iterable[float], typing.Iterable[float]], float]:

    def _minkowski_distance(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:
        dist = sum((
            abs(end_dim - start_dim) ** dims
            for (end_dim, start_dim)
            in zip(end, start)
        )) ** (1./dims)
        return dist

    if dims == float("-inf"):
        return antichebyshev_distance

    if dims == float("inf"):
        return chebyshev_distance

    return _minkowski_distance


def obstacle_dist(
    graph,
    base_measure=manhattan_distance
):

    def _actual_dist_func(start: typing.Iterable[float], end: typing.Iterable[float]) -> float:

        base_cost = base_measure(start, end)
        extra_cost = 0

        if graph[end] in (SLOW,):
            extra_cost += 100*base_cost

        total_cost = base_cost + extra_cost

        return total_cost

    return _actual_dist_func


def action_graph_dist(
    graph,
    default_cost=float("inf")
):

    def _actual_measure(start, end) -> float:
        cost = default_cost
        adjacents_gen = graph.adjacent_lazy(start)

        for adjacent in adjacents_gen:
            if adjacent == end:
                cost = graph[",".join((start, end))]

        return cost

    return _actual_measure

