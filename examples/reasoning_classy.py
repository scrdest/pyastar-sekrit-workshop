from goapystar.maputils import load_map_json
from goapystar.usecases.actiongraph.goap import FancyActionGraphGOAP
from goapystar.usecases.actiongraph.actiongraph import ActionGraph


def main():
    start = {"HasDirtyDishes": 1}
    goal = {"Debug": 1, "HasFood": 1}

    raw_map = load_map_json("map1")

    newmap = (
        ActionGraph(raw_map)
        .set_current(start)
        .set_goal(goal)
    )

    solver = FancyActionGraphGOAP(
        mapobj=newmap,
        cutoff_iter=500,
        max_queue_size=100,
    )

    cost, path = solver(
        start_pos=start,
        goal=goal,
    )

    print(cost)

    newmap.visualize()


if __name__ == '__main__':
    main()
