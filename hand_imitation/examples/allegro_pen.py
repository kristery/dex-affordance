from hand_imitation.env.environments.allegro_env.allegro_pen_dapg import PenEnvV0
import numpy as np

if __name__ == '__main__':
    env = PenEnvV0(has_renderer=True)
    env.seed(0)
    np.random.seed(0)
    obs = env.reset()
    low, high = env.action_spec

    # do visualization
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
        print(action)
        print(obs)
