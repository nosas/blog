import os

from env import GameEnv
from game import Game
from gym import wrappers
from stable_baselines3 import A2C, PPO

log_dir = "logs"
model_dir = "models/PPO10_random_spawns"
model_class = PPO
model_name = "PPO10_random_spawns_1"

# Create the log and model directories
for dir in [log_dir, model_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Create Game Environment for stable-baselines to utilize
# Flatten the observation space from a dict to a vectorized np.array
env = wrappers.FlattenObservation(GameEnv(game=Game()))
env.reset()

# Create the model
# model = model_class(policy="MlpPolicy", env=env, tensorboard_log=log_dir, verbose=1)
model = model_class.load(
    path=f"{model_dir}/455000.zip", env=env, tensorboard_log=log_dir, verbose=1
)

# Train the model for 2.5million timesteps
timesteps = 5000
episodes = 100

# Change Map at some timestep
maps = {
    timesteps: ("map_goal1_straight.tmx", 100),
    timesteps * 5: ("map_goal2_left.tmx", 100),
    timesteps * 15: ("map_goal3_right_down.tmx", 100),
    timesteps * 40: ("map_goal4_behind_wall.tmx", 100),
    timesteps * 65: ("map_goal5_end.tmx", 150),
}

for episode in range(1, episodes + 1):
    print("Episode", episode)
    env.reset()
    if timesteps * episode in maps:
        map_name, max_distance = maps[timesteps * episode]
        env.env.game._load_map(map_name=map_name)
        # env.set_max_distance(max_distance)
        env.reset()

    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{timesteps*episode}_1")

env.close()
