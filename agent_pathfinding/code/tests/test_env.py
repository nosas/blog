from game import Game
from env import GameEnv


class TestEnv:
    env = GameEnv(game=Game())

    def test_env_creation(self):
        assert self.env is not None

    def test_env_observation(self):
        obs = self.env.reset()
        assert type(obs) == dict
        assert len(obs) == 13

    def test_env_step(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(action=9)
        for var in [obs, reward, done, info]:
            assert var is not None

    def test_env_done(self):
        self.env.game._load_map(map_name="map_goal1_straight.tmx")
        _ = self.env.reset()

        for i in range(1000):
            new_obs, reward, done, _ = self.env.step(action=1)
            print(i, new_obs, reward)
            if done:
                break

        assert done
