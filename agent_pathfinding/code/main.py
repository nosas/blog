from game import Game

maps = {
    1: "map_goal1_straight.tmx",
    2: "map_goal2_left.tmx",
    3: "map_goal3_right_down.tmx",
    4: "map_goal4_behind_wall.tmx",
    5: "map_goal5_end.tmx",
}

if __name__ == "__main__":
    manual = 1
    g = Game(manual=manual, map_name=maps[2])

    if manual:
        while True:
            g.new()
            g.run()
            g.quit()
    else:
        import gym
        from env import GameEnv
        from stable_baselines3 import PPO

        genv = GameEnv(game=g)
        env = gym.wrappers.FlattenObservation(genv)

        model_class = PPO
        model_dir = "models/PPO3"
        model_path = f"{model_dir}/500000.zip"
        model = model_class.load(path=model_path)

        for _ in range(10):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation=obs)
                obs, reward, done, info = env.step(action=action)

        env.close()
