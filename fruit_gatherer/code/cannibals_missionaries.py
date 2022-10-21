"""
This program solves the following problem:

There are three cannibals and three missionaries on one side of the river.
There exists a boat in the river that may take one or two people across at a time.
At any time, the cannibals must not outnumber the missionaries or else the missionaries will be eaten.

The goal is to get all 6 of them to the other side of the river.
"""

# %%
from itertools import combinations
from copy import deepcopy
from queue import Queue


def is_fail(rivers: list[list[int]]):
    return any(
        [river.count(1) > river.count(0) and river.count(0) != 0 for river in rivers]
    )


def is_goal(rivers) -> bool:
    return len(rivers[1]) == 0 and not is_fail(rivers)


def get_actions(rivers: list[int], boat: str) -> list[tuple[int]]:
    def is_valid_action(action):
        result_rivers, _ = result(action, deepcopy(rivers), boat)
        return not is_fail(result_rivers)

    river = rivers[1] if boat == "right" else rivers[0]
    possible_actions = set(
        [
            tuple(sorted(t))
            for t in list(combinations(river, 2)) + list(combinations(river, 1))
        ]
    )
    actions = [a for a in possible_actions if is_valid_action(a)]
    return actions


def result(action: tuple[int, int], rivers: list[list[int]], boat: str):
    if boat == "right":
        to_river, from_river = rivers
    else:
        from_river, to_river = rivers

    for person in action:
        to_river.append(from_river.pop(from_river.index(person)))

    from_river = sorted(from_river)
    to_river = sorted(to_river)

    if boat == "right":
        result_rivers = [
            to_river,
            from_river,
        ]  # Ensure rivers always returns [left_river, right_river]
        boat = "left"
    else:
        result_rivers = [
            from_river,
            to_river,
        ]  # Ensure rivers always returns [left_river, right_river]
        boat = "right"

    return result_rivers, boat


print(is_fail([[1, 1, 0], [0, 0]]))  # True
print(is_fail([[1, 1, 1], [0, 1]]))  # False
print(is_fail([[1, 0, 0], [0, 0]]))  # False

print(is_goal([[0, 1, 0, 1], []]))  # True
print(is_goal([[0, 1, 0, 1, 1], []]))  # False
print(is_goal([[0, 1, 0, 1, 0], [1]]))  # False

# %%
r = [[[1, 1], [0, 0, 0, 1]], [[0, 1], [0, 1, 0, 1]], [[1, 1, 1], [0, 0, 0]]]


river_left = []
river_right = [0, 0, 0, 1, 1, 1]  # Cannibals are represented as 1, missionaries as 0

rivers = [river_left, river_right]
boat = "right"


# %%
rivers = r[1]
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
river_left = []
river_right = [0, 0, 1, 1]  # Cannibals are represented as 1, missionaries as 0

rivers = [river_left, river_right]
boat = "right"


def search(rivers, boat):
    reached = {(str(rivers), boat): [()]}
    frontier = Queue()
    for action in get_actions(rivers=rivers, boat=boat):
        frontier.put(action)

    while not frontier.empty():
        action = frontier.get()
        path = reached[(str(rivers), boat)]
        if is_goal(rivers):
            return path

        result_river, result_boat = result(action, deepcopy(rivers), boat)
        res = (str(result_river), result_boat)
        if res not in reached:
            reached[res] = path + [action]

search(rivers, boat)



# %%
