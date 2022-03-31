import os

from env import GameEnv
from game import Game
from gym import wrappers
from stable_baselines3 import A2C, PPO

log_dir = "logs/v2"
model_dir = "models/v2"
model_class = A2C
model_name = "A2C_Neo"
# old_model_name = "PPO"

# Create the log and model directories
for dir in [log_dir, model_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Create Game Environment for stable-baselines to utilize
# Flatten the observation space from a dict to a vectorized np.array
g = Game(rand_agent_spawn=1, rand_goal_spawn=1)
env = wrappers.FlattenObservation(GameEnv(game=g))
env.reset()

# Create the model
model = model_class(policy="MlpPolicy", env=env, tensorboard_log=log_dir, verbose=1)
# model = model_class.load(
#     path=f"{model_dir}/{old_model_name}/1250000.zip",
#     env=env,
#     tensorboard_log=log_dir,
#     verbose=1,
# )

# Train the model for 2.5million timesteps
timesteps = 2000
episodes = 1000

# Change Map at some timestep
maps = {1: ("map_train3.tmx", 100)}

for episode in range(1, episodes + 1):
    print("Episode", episode)
    env.reset()
    if episode in maps:
        map_name, max_distance = maps[episode]
        env.env.game._load_map(map_name=map_name)
        env.set_max_distance(max_distance)
        env.reset()

    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{model_name}/{timesteps*episode}")

env.close()
