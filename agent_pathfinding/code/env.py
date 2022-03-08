import numpy as np

import gym

from config import TILESIZE, WIDTH, HEIGHT
from math import sqrt
from sys import maxsize
from typing import Tuple


class GameEnv(gym.Env):
    def __init__(self, game) -> None:
        super().__init__()
        # [W, A, S, D, WA, WD, SA, SD, None/Neutral]
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
        self.observation_space = gym.spaces.Dict(
            {
                "dist_to_goal": gym.spaces.Box(
                    low=0,
                    high=sqrt(WIDTH**2 + HEIGHT**2) / TILESIZE,
                    dtype=np.float32,
                    shape=(1, 1),
                ),
                "dist_traveled": gym.spaces.Box(
                    low=0, high=np.inf, dtype=np.float32, shape=(1, 1)
                ),
                "heading": gym.spaces.Box(
                    low=0, high=360, dtype=np.float32, shape=(1, 1)
                ),
                "is_battling": gym.spaces.Discrete(2),
                "posx": gym.spaces.Box(
                    low=0, high=WIDTH / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "posy": gym.spaces.Box(
                    low=0, high=HEIGHT / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "goal_posx": gym.spaces.Box(
                    low=0, high=WIDTH / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "goal_posy": gym.spaces.Box(
                    low=0, high=HEIGHT / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "cardinal_objs": gym.spaces.Box(
                    low=-1, high=2, dtype=np.int8, shape=(1, 4)
                ),
                "cardinal_dists": gym.spaces.Box(
                    low=0, high=np.inf, dtype=np.float32, shape=(1, 4)
                ),
            }
        )

        self.game = game
        self._max_distance = maxsize

    @property
    def max_distance(self) -> int:
        return self._max_distance

    def calculate_reward(self, obs) -> float:
        if obs["is_battling"]:  # -1000 if Battle is not the Goal
            reward = -100
        elif obs["dist_to_goal"] < 0.7:  # +1000 if dist_to_goal < 0.7
            reward = 500
        else:  # Increase reward when Agent travels further distance
            reward = -1
            # If the Agent moved at least 0.1 of a tile
            if self.prev_dist_traveled - obs["dist_traveled"] > 0.5:
                self.prev_dist_traveled = obs["dist_traveled"]
                reward += 25
                if self.prev_dist_to_goal - obs["dist_to_goal"] >= 1:
                    reward += 50
            if self.prev_dist_to_goal - obs["dist_to_goal"] >= 1:
                self.prev_dist_to_goal = obs["dist_to_goal"]
                reward += 50
            else:
                reward -= 1
            if any([dist < 0.3 for dist in obs["cardinal_dists"]]):
                reward -= 20

        return reward

    def reset(self) -> float:
        """Set Game to clean state and return Agent's initial observation"""
        self.game.new()
        obs = self.game.agent.observation

        self.prev_dist_traveled = obs["dist_traveled"]
        self.prev_dist_to_goal = obs["dist_to_goal"]

        return obs

    def set_max_distance(self, max_distance: int) -> None:
        self._max_distance = max_distance

    def step(self, action) -> Tuple[float, float, bool, dict]:
        """Return observation (obj), reward (float), done (bool), info (dict)"""
        self.game._update(action=action)
        obs = self.game.agent.observation
        reward = self.calculate_reward(obs=obs)
        if obs["dist_traveled"] > self.max_distance:
            # TODO Detect if Agent is stuck in a corner for too long
            self.game.playing = False
        done = not self.game.playing
        info = {}

        return (obs, reward, done, info)
