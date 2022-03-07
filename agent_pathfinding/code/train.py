import os

from env import GameEnv
from game import Game
from gym import wrappers
from stable_baselines3 import PPO

log_dir = "logs"
model_dir = "models/PPO"
model_class = PPO
model_name = "PPO"

# Create the log and model directories
for dir in [log_dir, model_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Create Game Environment for stable-baselines to utilize
# Flatten the observation space from a dict to a vectorized np.array
env = wrappers.FlattenObservation(GameEnv(game=Game(map_name="map_goal1_straight.tmx")))
env.reset()

# Create the model
model = model_class(policy="MlpPolicy", env=env, tensorboard_log=log_dir, verbose=1)

# Train the model for 2.5million timesteps
timesteps = 25000
episodes = 100

# Change Map at some timestep
maps = {
    250000: "map_goal2_left.tmx",
    750000: "map_goal3_right_down.tmx",
    1250000: "map_goal4_behind_wall.tmx",
    2000000: "map_goal5_end.tmx",
}

for episode in range(1, episodes + 1):
    print("Episode", episode)
    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{timesteps*episode}")

    if timesteps * episode in maps:
        env.env.game._load_map(map_name=maps[timesteps*episode])
    env.reset()

env.close()
