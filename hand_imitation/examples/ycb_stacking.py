from hand_imitation.env.environments.ycb_stacking_env import YCBStacking
import numpy as np

if __name__ == '__main__':
    env = YCBStacking(has_renderer=True, friction=(5, 0.5, 1), object_scales=(1.5, 1),
                      object_names=("sugar_box", "mug"), solimp="0.98 0.999 0.001 0.1 6")
    env.seed(0)
    np.random.seed(0)
    obs = env.reset()
    low, high = env.action_spec

    # do visualization
    import time

    tic = time.time()
    env.sim.step()
    env.sim.forward()
    for i in range(10000):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
    env = None
    print(f"{time.time() - tic}s")
