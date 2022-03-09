import numpy as np
import pandas as pd

import gym

from collections import deque
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
                "is_headed_to_goal": gym.spaces.Discrete(2),
                "is_in_corner": gym.spaces.Discrete(2),
                "posx": gym.spaces.Box(
                    low=0, high=WIDTH / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "posy": gym.spaces.Box(
                    low=0, high=HEIGHT / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "delta_goal_posx": gym.spaces.Box(
                    low=-WIDTH / TILESIZE,
                    high=WIDTH / TILESIZE,
                    dtype=np.float32,
                    shape=(1, 1),
                ),
                "delta_goal_posy": gym.spaces.Box(
                    low=-HEIGHT / TILESIZE,
                    high=HEIGHT / TILESIZE,
                    dtype=np.float32,
                    shape=(1, 1),
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
        self._history_len = 100
        self._prev_obs = deque(maxlen=self._history_len)
        self._max_distance = maxsize

    @property
    def max_distance(self) -> int:
        return self._max_distance

    @property
    def _prev_obs_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._prev_obs)

    def calculate_reward(self, obs) -> float:
        self._prev_obs.append(obs)
        df = self._prev_obs_df
        reward = 0
        additional_reward = 0

        if obs["dist_to_goal"] < 0.7:
            reward += 100000 * max(0, (1 - obs["dist_traveled"] / self.max_distance))
        if df["is_in_corner"][-95:].sum() >= 95:
            print("Get outta the corner you dingus!")
            reward -= 50
        # if df["is_hitting_wall"]
        # if avg tile_position hasn't changed much in last 20-40 steps, punishment
        # if calculate_point_dist((int(posx), int(posy)), avg_tile_pos) < 5
        # self._prev_obs_df['dist_traveled'].diff(60)[-10:].round(2)

        # additional_reward = df["cardinal_objs"][:10].apply(lambda x: 1 in x).sum()
        additional_reward = 0
        if 1 in obs['cardinal_objs']:
            additional_reward += 2
        if obs['is_headed_to_goal']:
            additional_reward += 40
        slowed_down = self._prev_obs_df['dist_traveled'].diff(60)[-40:].mean() < 3
        if slowed_down:
            reward -= 1
        if any(obs["is_hitting_wall"]) and slowed_down:
            reward -= 10

        # Reward 0 to 14 for being close to the goal (within 100 tiles)
        if obs["dist_to_goal"] < 15:
            reward += (
                max(1, (100 - int(obs["dist_to_goal"]))) * max(1, additional_reward)
            ) / 100
        else:
            reward -= 1
        reward += additional_reward

        return reward

    def reset(self) -> float:
        """Set Game to clean state and return Agent's initial observation"""
        self.game.new()
        obs = self.game.agent.observation
        self._prev_obs = deque([obs] * self._history_len, maxlen=self._history_len)

        self.prev_dist_traveled = obs["dist_traveled"]
        self.prev_dist_to_goal = obs["dist_to_goal"]
        self.set_max_distance(max_distance=max(50, obs["dist_to_goal"] * 2))

        return obs

    def set_max_distance(self, max_distance: int) -> None:
        self._max_distance = int(max_distance)

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
