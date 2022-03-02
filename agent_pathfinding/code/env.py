import numpy as np

import gym


class GameEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # [W, A, S, D, WA, WS, SA, SD, None/Neutral]
        self.action_space = gym.spaces.Discrete(n=9)
        """
        ! NOTE This is incomplete and only a rough idea
        Observations include: [low, high]
        - Agent Positionx = [0, mob.pos.x/WIDTH] div by width to normalize to [0, 1]
        - Agent Positiony = [0, mob.pos.y/HEIGHT]
        - Agent Velocity = not necessary. Vel is static. Heading is important.
        - Agent Heading = [0, 360]
        - Nearest Mob Dist = [0, mob.distance/sqrt(WIDTH^2 + HEIGHT^2)]
        - Nearest Mob Posx = [0, mob.pos.y/WIDTH]
        - Nearest Mob Posy = [0, mob.pos.y/HEIGHT]
        - Nearest Goal Dist = range(0, goal.distance/sqrt(WIDTH^2 + HEIGHT^2))
        - Distance Moved = range(0, inf)
        """
        self.observation_space = gym.spaces.Box(
            low=np.array[0], high=np.array[np.inf], dtype=np.float32
        )

    def reset(self):
        pass

    def step(self):
        pass
