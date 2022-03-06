import os

from env import GameEnv
from game import Game
from stable_baselines3 import PPO

log_dir = "logs"
model_dir = "models/PPO_end"
model_class = PPO
model_name = "PPO_tp_end"

# Create the log and model directories
for dir in [log_dir, model_dir]:
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Create Game Environment for stable-baselines to utilize
env = GameEnv(game=Game())
env.reset()

# Create the model
model = model_class(
    policy="MultiInputPolicy", env=env, tensorboard_log=log_dir, verbose=1
)

# Train the model
timesteps = 20000
episodes = 50

for episode in range(1, episodes + 1):
    print("Episode", episode)
    model.learn(
        total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name=model_name
    )
    model.save(f"{model_dir}/{timesteps*episode}")

env.close()
