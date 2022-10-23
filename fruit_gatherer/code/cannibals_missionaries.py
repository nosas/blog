"""
This program solves the following problem:

There are three cannibals and three missionaries on one side of the river.
There exists a boat in the river that may take one or two people across at a time.
At any time, the cannibals must not outnumber the missionaries or else the missionaries will be eaten.

The goal is to get all 6 of them to the other side of the river.
"""

from collections import namedtuple
from copy import deepcopy

# %%
from itertools import combinations
from queue import Queue


def is_fail(rivers: str):
    """
    Return whether a state results in the missionaries being eaten

    Input:

    rivers (str): Example "cmmBccm" is a failed state, "cccmmmB" is the goal state
    """


    return any(
        [river.count('c') > river.count('m') and river.count('m') != 0 for river in rivers.split("B")]
    )


def is_goal(rivers: str) -> bool:
    return len(rivers.split("B")[1]) == 0 and not is_fail(rivers)


def get_actions(rivers: str, boat: str) -> list[tuple[int]]:
    def is_valid_action(action):
        result_rivers, _ = result(action, deepcopy(rivers), boat)
        return not is_fail(result_rivers)

    river = rivers.split("B")[1 if boat == "right" else 0]
    possible_actions = set(
        [
            tuple(sorted(t))
            for t in list(combinations(river, 2)) + list(combinations(river, 1))
        ]
    )
    actions = [a for a in possible_actions if is_valid_action(a)]
    return actions


def result(action: tuple[int, int], rivers: str, boat: str):
    def to_str(left, right):
        """Turn two lists of strings into a single string joined by 'B'"""
        return "B".join(["".join(sorted(left)), "".join(sorted(right))])

    left, right = list(map(list, rivers.split("B")))
    result_boat = "left" if boat == "right" else "right"

    if boat == "right":
        for person in action:
            left.append(right.pop(right.index(person)))
    else:
        for person in action:
            right.append(left.pop(left.index(person)))

    result_river = to_str(left, right)
    return result_river, result_boat


def search(rivers, boat):
    State = namedtuple("State", "rivers,boat")
    Action = namedtuple("Action", "action,state")

    state_init = State(rivers, boat)
    reached = {state_init: [()]}
    frontier = Queue()

    for action in get_actions(rivers=rivers, boat=boat):
        frontier.put(Action(action, state_init))

    while not frontier.empty():
        action = frontier.get()
        path = reached[action.state]
        if is_goal(action.state.rivers):
            return path[1:]

        result_river, result_boat = result(
            action.action, deepcopy(action.state.rivers), action.state.boat
        )
        state_res = State(result_river, result_boat)
        if state_res not in reached:  # Check for redundancies: turns tree search into graph search
            reached[state_res] = path + [action.action]
            for action in get_actions(rivers=state_res.rivers, boat=state_res.boat):
                frontier.put(Action(action, state_res))


# %% Search
river_left = ""
river_right = "mmmccc"  # Cannibals are represented as 1, missionaries as 0

rivers = "B".join([river_left, river_right])
boat = "right"

solution = search(rivers, boat)
print(solution)


# %% Scratch pad for tests
print(is_fail('ccmBcmm'))  # True
print(is_fail('ccBcmmm'))  # False
print(is_fail('cmmBmm'))  # False

print(is_goal('ccmmB'))  # True
print(is_goal('cccmmB'))  # False
print(is_goal('ccmmmBc'))  # False

# %%
rivers = 'Bcccmmm'
boat = "right"


# %%
a = get_actions(rivers, boat)
a

# %%
rivers1, boat1 = result(action=a[0], rivers=rivers, boat=boat)
rivers, boat, a[0], rivers1, boat1

# %%
a2 = get_actions(rivers1, boat)
a2
# %%
rivers2, boat2 = result(action=a2[0], rivers=rivers1, boat=boat1)
rivers1, boat1, a2[0], rivers1, boat2

# %%

# %%
