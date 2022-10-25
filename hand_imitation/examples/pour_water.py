from hand_imitation.env.environments.mug_pour_water_env import WaterPouringEnv
import numpy as np

if __name__ == '__main__':
    env = WaterPouringEnv(has_renderer=True)
    env.seed(0)
    np.random.seed(0)
    obs = env.reset()
    low, high = env.action_spec

    # do visualization
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
