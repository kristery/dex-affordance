import argparse
import glob
import os
import pickle

import cv2
import numpy as np

from hand_imitation.env.utils.mjcf_utils import xml_path_completion
from hand_imitation.kinematics.retargeting_optimizer import NaiveOptimizationRetargeting, \
    ChainMatchingPositionKinematicsRetargeting
from hand_imitation.misc.camera_utils import get_checkerboard_pose
from hand_imitation.misc.data_utils import load_hand_object_data, load_hand_object_data_v2


def parse_args():
    parser = argparse.ArgumentParser()
    # We assume that there is a directory called "object_pose" and a directory called "hand_pose" under root
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--task", type=str, default="placement")
    parser.add_argument("--object", type=str, default="banana")
    parser.add_argument("--object_cam", type=int, default=0)
    parser.add_argument("--hand_cam", type=str, default="multi_post")
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--icp", action="store_true", default=True)
    return parser.parse_args()


if __name__ == '__main__':
    # ["mug", "mustard_bottle", "tomato_soup_can", "sugar_box", "large_clamp"]
    args = parse_args()
    version = args.version
    np.set_printoptions(precision=4)
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")

    # Construct Kinematics Retargeting
    path = xml_path_completion("adroit/adroit_relocate.xml")
    if version < 2:
        link_names = ["palm", "thtip", "fftip", "mftip", "rftip", "lftip"]
        solver = NaiveOptimizationRetargeting(path, link_names, has_joint_limits=False)
        target_joint_index = [0, 4, 8, 12, 16, 20]
    elif 2 <= version < 5:
        link_names = ["palm", "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thtip", "fftip", "mftip",
                      "rftip", "lftip"][:6]
        solver = ChainMatchingPositionKinematicsRetargeting(path, link_names, has_joint_limits=True)
        target_joint_index = [0, 2, 6, 10, 14, 18, 4, 8, 12, 16, 20][:6]
    else:
        raise NotImplementedError

    # Calibrate camera and board
    calibration_image_path = os.path.join(project_root, "test_resources/calibration/v1-3/1_color/10.png")
    color_image = cv2.imread(calibration_image_path, cv2.IMREAD_COLOR)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    intrinsic = np.array([[612.99, 0, 317.723], [0, 613.113, 239.658], [0, 0, 1]])
    board2cam = get_checkerboard_pose(color_image, intrinsic=intrinsic, visualize=False, board_size=(4, 5),
                                      square_size=0.03)

    # Calibrate camera and table
    if version <= 3:
        cos, sin = np.cos(np.deg2rad(15)), np.sin(np.deg2rad(15))
        rot_board2world = np.array([[-cos, -sin, 0], [-sin, cos, 0], [0, 0, -1]]).T
        board2world = np.eye(4)
        board2world[:3, :3] = rot_board2world
        board2world[:3, 3] = [0.05, 0.06, 0.015]
    else:
        cos, sin = np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5))
        rot_board2world = np.array([[-cos, -sin, 0], [-sin, cos, 0], [0, 0, -1]]).T
        board2world = np.eye(4)
        board2world[:3, :3] = rot_board2world
        board2world[:3, 3] = [0.08, -0.03, 0.015]
    cam2world = board2world @ np.linalg.inv(board2cam)

    # Retargeting with task and object
    work_dir = os.path.join(args.data_root, "object_pose", args.task, args.object)
    obj_dirs = sorted(glob.glob(os.path.join(work_dir, "seq_*")))
    for obj_dir in obj_dirs:
        # obj_cam_dir = os.path.join(obj_dir, f"{args.object_cam}_objpose")
        obj_cam_dir = os.path.join(obj_dir, f"{args.object_cam}")
        hand_dir = obj_dir.replace("object_pose", "hand_pose")

        hand_cam_name = args.hand_cam
        if hand_cam_name == "1_post":
            hand_dir_name = "1_hand_post"
        else:
            hand_dir_name = f"{hand_cam_name}_hand"
        hand_cam_dir = os.path.join(hand_dir, hand_dir_name)

        retargeting_path = hand_cam_dir.replace("hand_pose", "retargeting")
        retargeting_path = retargeting_path.replace(hand_dir_name,
                                                    f"{args.hand_cam}_retargeting_v{args.version}.pkl")
        if not args.icp:
            obj_cam_dir = obj_cam_dir.replace(args.object, f"{args.object}_no_icp")
            retargeting_path = retargeting_path[:-4] + "_no_icp.pkl"
        if not args.overwrite and os.path.exists(retargeting_path):
            continue

        if version < 2:
            hand_seq, object_seq = load_hand_object_data(hand_cam_dir, obj_cam_dir, target_joint_index,
                                                         extrinsic=cam2world)
            robot_joints = solver.retarget(hand_seq, name=retargeting_path, verbose=False)
        else:
            if args.hand_cam == '0':
                hand_cam_to_object_cam = None
            else:
                if version <= 3:
                    hand_cam_to_object_cam = np.array(
                        [[-0.0178, -0.7087, -0.7053, 0.4976], [0.7386, 0.4662, -0.487, 0.3935],
                         [0.6739, -0.5296, 0.5151, 0.3154], [0., 0., 0., 1.]])
                else:
                    hand_cam_to_object_cam = np.array(
                        [[0.0454, 0.8483, 0.5276, -0.4907], [-0.8307, 0.3255, -0.4518, 0.3915],
                         [-0.5549, -0.4177, 0.7194, 0.3348], [0., 0., 0., 1.]])
                    hand_cam_to_object_cam = None

            hand_seq, object_seq, frame_seq, miss_num = load_hand_object_data_v2(hand_cam_dir, obj_cam_dir,
                                                                                 target_joint_index,
                                                                                 cam2world, hand_cam_to_object_cam)
            if miss_num > 30:
                continue

            robot_joints = solver.retarget(hand_seq, frame_seq, name=hand_cam_dir, verbose=False)

        result = {"retarget_qpos": robot_joints, "object_pose": object_seq}
        os.makedirs(os.path.dirname(retargeting_path), exist_ok=True)
        with open(retargeting_path, "wb") as f:
            pickle.dump(result, f)
