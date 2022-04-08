# First things first, import the GOAP class/function (whichever API you prefer).
# As mentioned, we'll use a ready-made variant of the algorithm tailored to NPC action graphs:
from goapystar.usecases.actions import ActionGOAP


def example():
    # 1. Define your action space as a map.
    #    - Keys are strings (Action names)
    #    - Values are 3-tuples: (cost, preconditions, effects)
    #      Cost is the abstract pathfinding 'distance' (you may choose to ignore it in custom heuristics).
    #      Preconditions will define when the Action is available.
    #      Effects will define what the planner *thinks* the Action will achieve
    #      (this may or may not be accurate, if you want fallible planners!).
    #
    #    For this example, let's write a simple, peaceful NPC living out his daily routine.
    #    This will let us use fairly natural, obvious and nontrivial sets of conditions/effects.
    raw_map = {
        'Eat': [
            1,
            {"HasFood": 1, "HasCleanDishes": 1},
            # By default, the result state is updated by adding old and new state and effect values together
            # so negative values represent using up resources and positives - acquiring them.
            {"HasDirtyDishes": 1, "HasCleanDishes": -1, "Fed": 1, "HasFood": -1}
        ],
        'Shop': [
            1,
            {"Money": 10},
            {"HasFood": 1, "Money": -10}
        ],
        'DishWash': [
            1,
            {"HasDirtyDishes": 1, "Rested": 1},
            {"HasDirtyDishes": -1, "HasCleanDishes": 1, "Rested": -1}
        ],
        'Work': [
            1,
            {"Rested": 1},
            {"Money": 10}
        ],
        'Idle': [
            1, 
            {},
            {"Rested": 1}
        ],
    }

    # 2. Define your starting state as a dict (can be empty).
    start = {"HasDirtyDishes": 1}

    # 3. Define your goal state.
    goal = {"Fed": 1, "HasCleanDishes": 1}

    # 4. When using the OOP API, instantiate our solver first
    #    In this case, we're using this API to avoid crafting
    #    custom callbacks for this pre-made use-case.
    solver = ActionGOAP(
        mapobj=raw_map,
        # We can limit max iterations - this is an NP problem, sometimes we just need to give up.
        # 50 iterations are enough for trivial problems
        # 500 iterations will do for our case, although it's still a fairly harsh constraint
        # 20k+ iterations is empirically a red flag that you should consider re-modelling the problem
        cutoff_iter=500,
        # We can also limit our priority queue size, but we won't (just arbitrarily).
        # However, setting a limit sometimes not only saves your RAM, but may also speed up the search!
        # None will set it to default (no limit).
        max_queue_size=None
    )

    # 5. Call our solver. The GOAP class is callable, so the usage conventions
    #    are common for the OOP and FP APIs.
    cost, path = solver(
        start_pos=start,
        goal=goal,
    )

    # 6. All done. Let's see the results!
    print(path)

    # 7. You should see:
    #    [State(_name='START', HasDirtyDishes=1), 'Idle', 'Work', 'DishWash', 'Shop', 'Eat']
    #    You can verify that this is solid - we wind up fed, with some dirty dishes.
    #
    #    Feel free to change the start/goal - just keep in mind sometimes there might not be a path at all!
    #    An obvious followup - try being tidy with the dishes - add '"HasCleanDishes": 1' to the Goal dict.
    return path


def test_example():
    example()
