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
env = wrappers.FlattenObservation(GameEnv(game=Game()))
env.reset()

# Create the model
model = model_class(
    policy="MlpPolicy", env=env, tensorboard_log=log_dir, verbose=1
)

# Train the model for 1million timesteps
timesteps = 25000
episodes = 40

for episode in range(1, episodes + 1):
    print("Episode", episode)
    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{timesteps*episode}")
    env.reset()

env.close()
