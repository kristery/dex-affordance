from hand_imitation.env.environments.dapg_env.relocate_v0 import RelocateEnvV0
import numpy as np

if __name__ == '__main__':
    env = RelocateEnvV0(has_renderer=True, primitive_name="sphere")
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
