# %%
from queue import PriorityQueue
from random import choice as rand_choice
from random import randrange as rand_range


def create_grid(size: int) -> str:
    """Given integer, return an input str"""
    if not (3 < size < 100):
        print("Invalid size, please input an integer N, where 0 < N < 100")
        return None

    # Create random fruit and agent position
    fruit_pos = agent_pos = (rand_range(0, size), rand_range(0, 5))
    while agent_pos[1] == fruit_pos[1]:
        agent_pos = (rand_range(0, 5), rand_range(0, 5))

    grid = []
    for i in range(size):
        row = "-" * size

        # Set the fruit's location
        if i == fruit_pos[0]:
            row = list(row)
            row[fruit_pos[1]] = "p"
            row = "".join(row)

        # Set the agent's location
        if i == agent_pos[0]:
            row = list(row)
            row[agent_pos[1]] = "a"
            row = "".join(row)

        grid.append(row)

    return "\n".join(grid)


print(create_grid(5))

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


def find_object(grid: list, symbol: str):
    for i, row in enumerate(grid):
        if symbol in set(row):
            for j, _ in enumerate(row):
                if grid[i][j] == symbol:
                    return (i, j)


def find_agent(grid):
    return find_object(grid, "a")


def find_goal(grid):
    return find_object(grid, "p")


def path_cost(grid: list, position: tuple):
    """Given a position on a grid, return the cost of the path to the goal"""

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


def nextMove(n,r,c,grid):
    g = [list(row) for row in grid]  # Convert to row of chars instead of strings
    agent_position = (r, c)
    solution = search(grid=g, grid_size=n, position=agent_position)
    return solution[0], solution



i = f"""
99
{create_grid(5)}
"""
m = int(i.split()[0])
grid = i.split()[1:]
r, c = (rand_range(0, 5), rand_range(0, 5))
g = [list(g) for g in grid]
print("\n".join(grid))

nextMove(m, r, c, grid)
