from typing import List, Dict
import numpy as np
import pickle
import os


def merge_demonstration(demo_list: List[Dict], fractions: List):
    assert len(demo_list) == len(fractions)

    demos = {}
    for demo_dict, fraction in zip(demo_list, fractions):
        demo_data = list(demo_dict.values())
        num_sample = int(len(demo_data) * fraction)

        k = 0
        for demo_name, single_demo_data in demo_dict.items():
            demos[demo_name] = single_demo_data
            k += 1
            if k > num_sample:
                break

    return demos


def main():
    data_root = "/home/sim/data/wanglab_dataset/dataset_v2/retargeting/relocation"
    demo_path_1 = data_root + "/mug/cammulti_post_all_demonstration_relocation_mug_v3.pkl"
    demo_path_2 = data_root + "/tomato_soup_can/cammulti_post_all_demonstration_relocation_tomato_soup_can_v3.pkl"
    demo_list = []
    for demo_path in [demo_path_1, demo_path_2]:
        demo_list.append(np.load(demo_path, allow_pickle=True))
    demos = merge_demonstration(demo_list, [1., 1.])
    os.makedirs(f"{data_root}/mixed", exist_ok=True)
    with open(f"{data_root}/mixed/cammulti_post_all_demonstration_relocation_mixed_v3.pkl", "wb") as f:
        pickle.dump(demos, f)


if __name__ == '__main__':
    main()
