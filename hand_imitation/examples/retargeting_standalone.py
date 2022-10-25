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
    parser.add_argument("-j", "--joint_file", type=str)
    parser.add_argument("-f", "--frame_file", type=str)
    parser.add_argument("-o", "--object_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--object_name", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=4)
    path = xml_path_completion("adroit/adroit_relocate.xml")
    link_names = ["palm", "thmiddle", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle", "thtip", "fftip", "mftip",
                  "rftip", "lftip"][:6]
    solver = ChainMatchingPositionKinematicsRetargeting(path, link_names, has_joint_limits=True,
                                                        has_global_pose_limits=False)
    target_joint_index = [0, 2, 6, 10, 14, 18, 4, 8, 12, 16, 20][:6]

    object_name = args.object_name
    joint_seq = np.load(args.joint_file)
    frame_seq = np.load(args.frame_file)
    object_seq = np.load(args.object_file)
    robot_joints = solver.retarget(joint_seq[:, target_joint_index], frame_seq, name="stand_alone", verbose=False)
    result = {"retarget_qpos": robot_joints, "object_pose": [{object_name: object_pose} for object_pose in object_seq]}

    with open(args.output_file, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()
