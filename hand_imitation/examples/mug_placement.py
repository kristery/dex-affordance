from hand_imitation.env.environments.mug_place_object_env import MugPlaceObjectEnv
import numpy as np
import time

if __name__ == '__main__':

    env = MugPlaceObjectEnv(has_renderer=True, friction=(1, 0.5, 0.01), object_scale=0.8, mug_scale=1.3, large_force=False)
    # env.seed(0)
    # np.random.seed(0)
    obs = env.reset()
    low, high = env.action_spec

    # do visualization
    tic = time.time()
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        # if i % 10 == 0:
        #     print(f"Percentage, {env.compute_intersection_rate()}")
        env.render()
        print(reward)
    print(f"Time: {time.time() - tic}")
    print("reset")
    env.reset()
    env.data.qpos[30:33] = [0, 0, 0.15]
    env.data.qpos[33:37] = [0.707, 0.707, 0, 0]
    # for i in range(300):
    #     action = np.random.uniform(low, high)
    #     obs, reward, done, _ = env.step(action)
    #     env.render()
    #     print(reward)
    # env = None
