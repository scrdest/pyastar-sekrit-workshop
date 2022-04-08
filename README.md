GoaPyStar
=========
#### Search-based planning AI for Python - lightweight, powerful and flexible

GoaPyStar is a Python implementation of the Goal-Oriented Action Planning (GOAP) algorithm.

GOAP is a STRIPS-like, search-based planning algorithm typically used for game AIs, 
originally implemented by Jeff Orkin for the 2005 videogame F.E.A.R. and subsequently 
featured in several other AAA releases.

Unlike the more traditional approaches that rely on the programmer to implement
sane transitions between actions available to the agent, GOAP takes a sequence of
actions (with preconditions for their execution and postconditions for their effects
defined) available to the agent and finds its own way into a solution using the
classic A* algorithm over the graph of actions. The result is an almost *notoriously* 
'smart' and somewhat unpredictable AI agent.

`> Wasn't this implemented for Python already? Why bother?` 

While there *are* a handful of existing implementations for Python, most of them seem to
be more of an academic exercise than a serious attempt at a Python library. 

GoaPyStar aims to fill that niche.


### o Selling points:

- **Lightweight**:
  - Written in pure Python with no third-party dependencies.
  - Algorithm can be constrained on both iterations and memory consumption.


- **Powerful**:
  - Handles more than your basic boolean tags. 
  - Supports full STRIPS-like state transitions, but uses standard Python syntax.
  - This also means complex plans, with back-and-forth state transitions, are fully supported!


- **Flexible**:
  - Will never force you to use custom data structures to run the algorithm. 
    States and goals are plain old dicts.
  - Functional and OOP APIs - use whichever you prefer!
  - Customizable:
    - Want your effects to overwrite the state? Add to it? Multiply negatives and add positives? **Can do!**
    - Want your own heuristics or distance measures? **Can do!**
    - Want available actions to change based on the current state? **Can do!**
    - Want to use the algorithm for a classic Astar path search? **Can do!**
      - *Wait, you meant in 3D?* **Can do!**
    - Just write a function that handles things the way you like and plug it in!
    

## Installation:

#### PIP + PyPI (recommended):

`pip install goapystar`

###### PIP + GitHub:

`pip install git+https://github.com/scrdest/pyastar-sekrit-workshop.git`

#### Clone (required for running examples/tests):

`git clone https://github.com/scrdest/pyastar-sekrit-workshop.git`

## Usage: 

For NPC actions, you can use the `goapystar.usecases.actions.ActionGOAP` class from
the OOP API - it comes prebuilt with methods to handle such use-cases.

If that doesn't suit you, customize away!

### Example:
```python
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


# Run the example:
if __name__ == '__main__':
  example()
```
