from game import Game

if __name__ == "__main__":
    manual = 1
    g = Game(manual=manual)

    if manual:
        while True:
            g.new()
            g.run()
            g.quit()
    else:
        from env import GameEnv
        from stable_baselines3 import PPO

        env = GameEnv(game=g)

        model_class = PPO
        model_dir = "models/PPO"
        model_path = f"{model_dir}/200000.zip"
        model = model_class.load(path=model_path)

        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs)
            obs, reward, done, info = env.step(action=action)

        env.close()
