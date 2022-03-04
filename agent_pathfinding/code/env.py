import numpy as np

import gym

from config import TILESIZE
from typing import Tuple


class GameEnv(gym.Env):
    def __init__(self, game) -> None:
        super().__init__()
        # [W, A, S, D, WA, WD, SA, SD, None/Neutral]
        self.action_space = gym.spaces.Discrete(n=9, start=1)
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
            low=np.array([0]), high=np.array([np.inf]), dtype=np.float32
        )

        self.game = game

    def calculate_reward(self, obs) -> float:
        return obs / TILESIZE / 1.3

    def render(self) -> None:
        self.game._draw()

    def reset(self) -> float:
        """Set Game to clean state and return Agent's initial observation"""
        self.game.new()
        return self.game.agent.distance_traveled

    def step(self, action) -> Tuple[float, float, bool, dict]:
        """Return observation (obj), reward (float), done (bool), info (dict)"""
        # action = self.action_space.sample()  # Select random action
        self.game._update(action=action)
        obs = self.game.agent.distance_traveled
        reward = self.calculate_reward(obs)
        done = not self.game.playing
        info = {}

        return (obs, reward, done, info)
