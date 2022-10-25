from hand_imitation.env.environments.ycb_relocate_env import YCBRelocate
import numpy as np

if __name__ == '__main__':
    original_names = ['sugar_box']
    transfer_object_names = ['sugar_box']
    transfer_object_scale = [1.0]

    for object_name, object_scale in zip(transfer_object_names, transfer_object_scale):
        env = YCBRelocate(has_renderer=True, object_name=object_name, friction=(1, 0.5, 0.01),
                          object_scale=object_scale, version="xml")
        env.seed(0)
        np.random.seed(0)
        obs = env.reset()
        low, high = env.action_spec

        # do visualization
        for i in range(100):
            action = np.random.uniform(low, high)
            obs, reward, done, _ = env.step(action)
            env.render()
            print(action)
            print(obs)
        env = None
