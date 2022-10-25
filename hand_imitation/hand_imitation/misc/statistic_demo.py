import os
import numpy as np
import tabulate

TASKS = ["relocation"] * 5 + ["pour_water"] + ["stacking"]
OBJECTS = ["mug", "sugar_box", "large_clamp", "tomato_soup_can", "mustard_bottle"] + ["mug"] + ["all"]


def main():
    root_dir = "/home/sim/data/wanglab_dataset/dataset_v2/retargeting"
    version = "3"
    use_icp = False
    headers = ["use_icp", "hand_setting", "task", "object", "num", "list"]
    table = []
    for task, object_name in zip(TASKS, OBJECTS):
        for camera_name in ["multi_post", "multi", "1", "1_post"]:
            demo_path = os.path.join(root_dir, task, object_name,
                                     f"cam{camera_name}_all_demonstration_{task}_{object_name}_v{version}.pkl")
            if not use_icp:
                demo_path = demo_path[:-4] + "_no_icp.pkl"
            if os.path.exists(demo_path):
                demo = np.load(demo_path, allow_pickle=True)
                demo_keys = list(demo.keys())
                demo_keys = [key.split("/")[0] for key in demo_keys]
                num_demo = len(demo_keys)
            else:
                demo_keys = []
                num_demo = 0

            table_row = [use_icp, camera_name, task, object_name, num_demo, demo_keys]
            table.append(table_row)

    print(tabulate.tabulate(table, headers=headers, tablefmt='tsv', floatfmt='.4f'))


if __name__ == '__main__':
    main()
