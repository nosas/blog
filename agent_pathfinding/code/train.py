import os

from env import GameEnv
from game import Game
from gym import wrappers
from stable_baselines3 import A2C, PPO

log_dir = "logs"
model_dir = "models"
model_class = PPO
model_name = "PPO31_full_map"
# old_model_name = "PPO22_GARBO"

# Create the log and model directories
for dir in [log_dir, model_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Create Game Environment for stable-baselines to utilize
# Flatten the observation space from a dict to a vectorized np.array
g = Game(rand_agent_spawn=0, rand_goal_spawn=0)
env = wrappers.FlattenObservation(GameEnv(game=g))
env.reset()

# Create the model
model = model_class(policy="MlpPolicy", env=env, tensorboard_log=log_dir, verbose=1)
# model = model_class.load(
#     path=f"{model_dir}/{old_model_name}/45000.zip",
#     env=env,
#     tensorboard_log=log_dir,
#     verbose=1,
# )

# Train the model for 2.5million timesteps
timesteps = 10000
episodes = 100

# Change Map at some timestep
maps = {timesteps: ("map_goal5_end.tmx", 150)}

for episode in range(1, episodes + 1):
    print("Episode", episode)
    env.reset()
    if timesteps * episode in maps:
        map_name, max_distance = maps[timesteps * episode]
        env.env.game._load_map(map_name=map_name)
        env.set_max_distance(max_distance)
        env.reset()

    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{model_name}/{timesteps*episode}")

env.close()
