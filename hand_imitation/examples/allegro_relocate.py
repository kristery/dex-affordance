import numpy as np

from hand_imitation.env.environments.allegro_env.allegro_relocate_dapg import RelocateEnvV0

if __name__ == '__main__':
    env = RelocateEnvV0(has_renderer=True, primitive_name="sphere", simple=False)
    env.seed(5)
    np.random.seed(5)
    obs = env.reset()
    low, high = env.action_spec

    # do visualization
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()

    env.reset()
    for i in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, _ = env.step(action)
        env.render()
