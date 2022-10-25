import argparse
import glob
import os
import pickle
import warnings

import numpy as np
from hand_imitation.kinematics.demonstration.pour_water_demo import WaterPouringDemonstration
from hand_imitation.kinematics.demonstration.relocation_demo import RelocationDemonstration
from hand_imitation.kinematics.demonstration.stacking_demo import StackingDemonstration
from hand_imitation.kinematics.demonstration.placement_demo import PlacementDemonstration


def parse_args():
    parser = argparse.ArgumentParser()
    # We assume that there is a directory called "object_pose" and a directory called "hand_pose" under root
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--task", type=str, default="placement")
    parser.add_argument("--object", type=str, default="banana")
    parser.add_argument("--hand_cam", type=str, default="multi_post")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--icp", action="store_true", default=True)
    return parser.parse_args()


if __name__ == '__main__':
    # ["mug", "mustard_bottle", "tomato_soup_can", "sugar_box", "large_clamp"]
    args = parse_args()
    np.set_printoptions(precision=4)
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

    if args.task == "relocation":
        demonstration_gen = RelocationDemonstration(has_renderer=False, object_name=args.object)
    elif args.task == "pour_water":
        demonstration_gen = WaterPouringDemonstration(has_renderer=False)
    elif args.task == "stacking":
        demonstration_gen = StackingDemonstration(has_renderer=False, object_names=["sugar_box", "mug", "large_clamp"],
                                                  object_scales=[1, 1, 1])
    elif args.task == "placement":
        demonstration_gen = PlacementDemonstration(has_renderer=False)

    else:
        raise ValueError(f"Task can only be relocation or reorientation, but given {args.task}")

    # Generate Demonstration based on Retargeting Result
    work_dir = os.path.abspath(os.path.join(args.data_root, "retargeting", args.task, args.object))
    work_dir = os.path.normpath(work_dir)
    obj_dirs = sorted(glob.glob(os.path.join(work_dir, "seq_*")))
    all_data = {}
    for obj_dir in obj_dirs:
        retargeting_path = os.path.join(obj_dir, f"{args.hand_cam}_retargeting_v{args.version}.pkl")
        if not args.icp:
            retargeting_path = retargeting_path[:-4] + "_no_icp.pkl"
        if not os.path.exists(retargeting_path):
            warnings.warn(f"Skip file {retargeting_path}")
            continue
        sub_path = os.path.normpath(retargeting_path)[len(work_dir) + 1:]
        data = np.load(retargeting_path, allow_pickle=True)
        demo_data = demonstration_gen.play_hand_object_seq(data["retarget_qpos"], data["object_pose"], sub_path)
        if demo_data is None:
            continue
        demo_path = retargeting_path.replace("_retargeting_", "_demonstration_")
        with open(demo_path, "wb") as f:
            pickle.dump(demo_data, f)
        all_data[sub_path] = demo_data

    all_demo_path = os.path.join(work_dir,
                                 f"cam{args.hand_cam}_all_demonstration_{args.task}_{args.object}_v{args.version}.pkl")
    if not args.icp:
        all_demo_path = all_demo_path[:-4] + "_no_icp.pkl"
    with open(all_demo_path, "wb") as f:
        pickle.dump(all_data, f)
