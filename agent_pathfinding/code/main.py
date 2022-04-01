from game import Game
from goals import Goal

maps = {
    0: "map_train1.tmx",
    1: "map_goal1_straight.tmx",
    2: "map_goal2_left.tmx",
    3: "map_goal3_right_down.tmx",
    4: "map_goal4_behind_wall.tmx",
    5: "map_goal5_end.tmx",
    6: "map_test_sensor.tmx",
}

if __name__ == "__main__":
    manual = True
    # g = Game(manual=manual, map_name=maps[2])
    g = Game(
        manual=manual,
        map_name=maps[0],
        rand_agent_spawn=False,
        rand_goal_spawn=False,
        draw_info_boxes=False,
    )

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
        model_dir = "models/PPO27_genius"
        model_path = f"{model_dir}/940000.zip"
        model = model_class.load(path=model_path)

        for _ in range(10):
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation=obs)
                obs, reward, done, info = env.step(action=action)

        env.close()
