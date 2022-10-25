import os
import numpy as np
import argparse
import transforms3d
import csv

TASKS = [("relocation", "mug"), ("relocation", "sugar_box"), ("relocation", "tomato_soup_can"),
         ("relocation", "mustard_bottle"), ("relocation", "large_clamp"), ("pour_water", "mug"), ("stacking", "misc")][
        :-1]

METRICS = ["mean", "min", "max", "std"]
ENTRY_NAMES = ["init_pos", "target_pos", "reward", "action"]


def parse_args():
    parser = argparse.ArgumentParser()
    # We assume that there is a directory called "object_pose" and a directory called "hand_pose" under root
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--hand_cam", type=str, default="multi_post")
    parser.add_argument("--version", type=int, default=3)
    return parser.parse_args()


def collect_object_poses(sim_data):
    qpos = sim_data[0]['qpos']
    object_qpos = qpos[30:37]
    axis, angle = transforms3d.quaternions.quat2axangle(object_qpos[3:])
    return object_qpos[:3], axis, angle


def collect_single_demonstration_file(demo_path, object_name, task_name):
    object_positions = []
    target_positions = []
    actions = []
    rewards = []
    all_data = np.load(demo_path, allow_pickle=True)
    for data_name, demonstration in all_data.items():
        observation = demonstration['observations']
        reward = demonstration['rewards']
        action = demonstration['actions']
        sim_data = demonstration['sim_data']
        object_pos = sim_data[0]['qpos'][30:32]
        object_target = sim_data[-1]['qpos'][30:33]
        object_positions.append(object_pos)
        target_positions.append(object_target)
        actions.append(np.mean(action, axis=0))
        rewards.append(np.mean(reward, axis=0))

    task_row = [task_name, object_name]
    action_row = []
    reward_row = []
    for metric, array in zip(ENTRY_NAMES[:2], [object_positions, target_positions]):
        data_array = np.array(array)
        task_row.append(np.array2string(np.mean(data_array, axis=0), separator=","))
        task_row.append(np.array2string(np.min(data_array, axis=0), separator=","))
        task_row.append(np.array2string(np.max(data_array, axis=0), separator=","))
        task_row.append(np.array2string(np.std(data_array, axis=0), separator=","))

    for metric, array in zip(ENTRY_NAMES[2:3], [rewards]):
        data_array = np.array(array)
        reward_row.append(np.array2string(np.mean(data_array, axis=0), separator=","))
        reward_row.append(np.array2string(np.min(data_array, axis=0), separator=","))
        reward_row.append(np.array2string(np.max(data_array, axis=0), separator=","))
        reward_row.append(np.array2string(np.std(data_array, axis=0), separator=","))

    for metric, array in zip(ENTRY_NAMES[3:4], [actions]):
        data_array = np.array(array)
        action_row.append(np.array2string(np.mean(data_array, axis=0), separator=","))
        action_row.append(np.array2string(np.min(data_array, axis=0), separator=","))
        action_row.append(np.array2string(np.max(data_array, axis=0), separator=","))
        action_row.append(np.array2string(np.std(data_array, axis=0), separator=","))

    return task_row, reward_row, action_row


if __name__ == '__main__':
    from hand_imitation.misc.path_utils import get_project_root

    args = parse_args()
    np.set_printoptions(precision=4)
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
    header = ["task", "object"]
    for entry in ENTRY_NAMES:
        for metric in METRICS:
            header.append(f"{entry}({metric})")

    with open(os.path.join(get_project_root(), "summary", f"{args.hand_cam}_v{args.version}.csv"), "w") as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(header)
        for task_tuple in TASKS:
            task_name, object_name = task_tuple
            work_dir = os.path.abspath(os.path.join(args.data_root, "retargeting", task_name, object_name))
            work_dir = os.path.normpath(work_dir)
            demo_file = f"cam{args.hand_cam}_all_demonstration_{task_name}_{object_name}_v{args.version}.pkl"
            demo_path = os.path.join(work_dir, demo_file)
            task_info, reward_info, action_info = collect_single_demonstration_file(demo_path, object_name, task_name)
            csv_writer.writerow(task_info + reward_info + action_info)
