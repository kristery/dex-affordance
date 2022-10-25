import os
import numpy as np
from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration
from hand_imitation.env.utils.gif_recording import RecordingGif

TASKS = ["relocation"] * 5 + ["pour_water"] + ["stacking"]
OBJECTS = ["mug", "sugar_box", "large_clamp", "tomato_soup_can", "mustard_bottle"] + ["mug"] + ["all"]


def main():
    root_dir = "/home/sim/data/wanglab_dataset/dataset_v2/retargeting"
    version = "3"
    use_icp = True
    for task, object_name in zip(TASKS, OBJECTS):
        if task is not "relocation":
            continue

        env = RelocationDemonstration(has_renderer=False, object_name=object_name)
        for camera_name in ["multi_post"]:
            demo_path = os.path.join(root_dir, task, object_name,
                                     f"cam{camera_name}_all_demonstration_{task}_{object_name}_v{version}.pkl")
            if not use_icp:
                demo_path = demo_path[:-4] + "_no_icp.pkl"
            if os.path.exists(demo_path):
                demo = np.load(demo_path, allow_pickle=True)
                demo_keys = list(demo.keys())
                demo_seqs = [key.split("/")[0] for key in demo_keys]
                for demo_key, demo_seq in zip(demo_keys, demo_seqs):
                    video_path = os.path.join(demo_path.replace("retargeting", "demo_video")[:-4], demo_seq)
                    with RecordingGif(env, save_directory=video_path, freq=25) as gif:
                        env.replay_state(demo[demo_key])
            else:
                continue


if __name__ == '__main__':
    main()
