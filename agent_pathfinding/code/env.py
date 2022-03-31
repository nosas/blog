import numpy as np
import pandas as pd

import gym

from collections import deque
from config import TILESIZE, WIDTH, HEIGHT
from math import sqrt
from typing import Tuple


class GameEnv(gym.Env):
    def __init__(self, game) -> None:
        super().__init__()
        # [W, A, S, D, WA, WD, SA, SD, None/Neutral]
        self.action_space = gym.spaces.Discrete(n=9)
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
                "heading_goal": gym.spaces.Box(
                    low=0, high=360, dtype=np.float32, shape=(1, 1)
                ),
                "is_battling": gym.spaces.Discrete(2),
                "is_headed_to_goal": gym.spaces.Discrete(2),
                "is_hitting_wall": gym.spaces.Discrete(2),
                "is_reached_goal": gym.spaces.Discrete(2),
                "tposx": gym.spaces.Box(
                    low=0, high=WIDTH / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "tposy": gym.spaces.Box(
                    low=0, high=HEIGHT / TILESIZE, dtype=np.float32, shape=(1, 1)
                ),
                "delta_goal_tposx": gym.spaces.Box(
                    low=-WIDTH / TILESIZE,
                    high=WIDTH / TILESIZE,
                    dtype=np.float32,
                    shape=(1, 1),
                ),
                "delta_goal_tposy": gym.spaces.Box(
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
        self._max_distance = 75

    @property
    def max_distance(self) -> int:
        return self._max_distance

    @property
    def _prev_obs_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._prev_obs)

    def calculate_reward(self, obs) -> float:
        def collision():
            """Calculate the reward given when the Agent collides"""
            if obs["is_hitting_wall"]:
                print("Stop hitting walls!")
                return -10
            return 0

        def goal() -> int:
            """Calculate the reward given when the Agent reaches the Goal or moves"""
            if obs["is_reached_goal"]:
                print(
                    f"Reached Goal in {int(obs['dist_traveled'])}/{self.max_distance} tiles"
                )
                return 10
            return self._prev_obs[-1]["dist_to_goal"] - obs["dist_to_goal"]

        rew_coll = collision()
        rew_goal = goal()
        rew_total = rew_coll + rew_goal

        return rew_total

    def reset(self) -> float:
        """Set Game to clean state and return Agent's initial observation"""
        self.game.new()
        obs = self.game.agent.observation
        self._prev_obs = deque([obs] * self._history_len, maxlen=self._history_len)

        self.prev_dist_traveled = obs["dist_traveled"]
        self.prev_dist_to_goal = obs["dist_to_goal"]

        return obs

    def set_max_distance(self, max_distance: int) -> None:
        self._max_distance = int(max_distance)

    def step(self, action) -> Tuple[float, float, bool, dict]:
        """Return observation (obj), reward (float), done (bool), info (dict)"""
        self.game._update(action=action)
        obs = self.game.agent.observation
        reward = self.calculate_reward(obs=obs)
        self._prev_obs.append(obs)

        termination = [
            obs["dist_traveled"] > self.max_distance,
            obs["is_reached_goal"],
            obs["is_hitting_wall"],
        ]

        if any(termination):
            self.game.playing = False
        done = not self.game.playing
        info = {}

        return (obs, reward, done, info)
