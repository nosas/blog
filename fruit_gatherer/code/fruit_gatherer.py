# %%
from queue import PriorityQueue
from random import choice as rand_choice


def create_grid(size: int) -> str:
    """Given an odd integer, return an input str"""
    if size % 2 == 0 or (size < 3 or size > 99):
        print("Invalid size, please input an odd integer between 3 and 99")
        return None

    corners = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
    corner = rand_choice(corners)

    grid = []
    for i in range(size):
        row = "-" * size

        # Set the fruit's location
        if i == corner[0]:
            row = list(row)
            row[corner[1]] = "f"
            row = "".join(row)

        # Set the agent's location
        if i == size // 2:
            row = list(row)
            row[size // 2] = "a"
            row = "".join(row)
        grid.append(row)

    return "\n".join(grid)


# %%
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]


def is_goal(grid: list, position: tuple):
    return grid[position[0]][position[1]] == "p"


def get_result(action: str, position: tuple):
    """Transition model"""
    if action == "UP":  # Decrease y-value by 1
        new_pos = (position[0] - 1, position[1])
    elif action == "DOWN":  # Increase y-value by 1
        new_pos = (position[0] + 1, position[1])
    elif action == "LEFT":  # Decrease x-value by 1
        new_pos = (position[0], position[1] - 1)
    elif action == "RIGHT":  # Increase x-value by 1
        new_pos = (position[0], position[1] + 1)
    return new_pos


def get_actions(grid: list, position: tuple):
    grid_size = len(grid)

    def is_valid_action(action: str):
        new_pos = get_result(action, position)
        # Verify new position is within the grid's bounds
        for pos in new_pos:
            if not (0 <= pos < grid_size):
                return False
        return True

    possible_actions = [action for action in ACTIONS if is_valid_action(action)]
    return possible_actions


def path_cost(grid: list, position: tuple):
    """Given a position on a grid, return the cost of the path to the goal"""

    def find_goal(grid) -> tuple:
        size = len(grid)
        corners = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
        for corner in corners:
            if is_goal(grid, corner):
                return corner

    goal_pos = find_goal(grid)
    path_cost = abs(goal_pos[0] - position[0]) + abs(goal_pos[1] - position[1])
    return path_cost


def search(grid, grid_size: int, position: tuple, path: list = None):

    # Calculate the distance to the goal from every position in the grid
    costs = {}
    for i in range(grid_size):
        for j in range(grid_size):
            costs[(i, j)] = path_cost(grid, (i, j))

    frontier = PriorityQueue()
    frontier.put((costs[position], position))
    reached = {position: []}

    while not frontier.empty():
        _, pos = frontier.get()
        path = reached[pos]
        if is_goal(grid, position=pos):
            return path

        # Get possible actions from pos
        for action in get_actions(grid=grid, position=pos):
            new_pos = get_result(action=action, position=pos)
            if new_pos not in reached:
                reached[new_pos] = path + [action]
                frontier.put((costs[new_pos], new_pos))

    return path


def displayPathtoPrincess(n: int, grid: list):
    # print all the moves here
    g = [list(row) for row in grid]  # Convert to row of chars instead of strings
    agent_position = (n // 2, n // 2)
    solution = search(grid=g, grid_size=n, position=agent_position)
    print("\n".join(solution))

i = """
3
---
-m-
p--
"""
m = int(i.split()[0])
grid = i.split()[1:]
# m = int(input())
# grid = create_grid(size=m).split("\n")
g = [list(g) for g in grid]
print(g)

displayPathtoPrincess(m, grid)

# %%
def find_goal(grid) -> tuple:
    size = len(grid)
    corners = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
    for corner in corners:
        if is_goal(grid, corner):
            return corner