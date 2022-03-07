from game import Game

if __name__ == "__main__":
    manual = 0
    g = Game(manual=manual)

    if manual:
        while True:
            g.new()
            g.run()
            g.quit()
    else:
        import gym
        from env import GameEnv
        from stable_baselines3 import PPO

        env = gym.wrappers.FlattenObservation(GameEnv(game=g))

        model_class = PPO
        model_dir = "models/PPO1_straight_line"
        model_path = f"{model_dir}/100000.zip"
        model = model_class.load(path=model_path)

        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs)
            obs, reward, done, info = env.step(action=action)

        env.close()
