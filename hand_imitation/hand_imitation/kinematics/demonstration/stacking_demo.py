import time
from typing import List, Dict
import warnings

import numpy as np
import transforms3d
from hand_imitation.env.environments.ycb_stacking_env import YCBStacking
from hand_imitation.kinematics.demonstration.base import DemonstrationBase, LPFilter
from hand_imitation.misc.data_utils import interpolate_replay_sequence, min_jerk_interpolate_replay_sequence
from hand_imitation.env.models.objects import YCB_SIZE


class StackingDemonstration(DemonstrationBase, YCBStacking):
    def __init__(self, has_renderer, **kwargs):
        super().__init__(has_renderer, -1, **kwargs)
        self.filter = LPFilter(30, 5)
        self.init_sim_data = self.dump()
        self.init_model_data = self.dump_mujoco_model()
        object_qpos_indices = [self.get_object_joint_qpos_indices(object_name) for object_name in
                               self.object_names[1:3]]
        self.object_trans_qpos_indices = [object_qpos_index[:3] for object_qpos_index in object_qpos_indices]
        self.object_rot_qpos_indices = [object_qpos_index[3:] for object_qpos_index in object_qpos_indices]

    def play_hand_object_seq(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]],
                             name="undefined"):
        tic = time.time()

        # Remove clamp data with only 2 objects
        if len(self.object_names) < 3:
            for object_pose_dict in object_pose_seq:
                if "large_clamp" in object_pose_dict:
                    object_pose_dict.pop("large_clamp")

        # Processing Sequence Data
        reference_object_name = self.object_names[0]
        action_time_step = self.control_timestep
        collect_time_step = 1 / 25.0
        retarget_qpos_seq, object_pose_seq = self.strip_stacking(retarget_qpos_seq, object_pose_seq)
        if reference_object_name not in object_pose_seq[-1]:
            object_pose_seq = object_pose_seq[:-1]
            retarget_qpos_seq = retarget_qpos_seq[:-1]
        if object_pose_seq[-1][reference_object_name][2, 3] < -0.02:
            warnings.warn("Target position has a z < -0.02, skip it!")
            return None

        init_object_lift = object_pose_seq[-1]["sugar_box"][2, 3] - YCB_SIZE["sugar_box"][0]
        retarget_qpos_seq, object_pose_seq = self.hindsight_replay_sequence(retarget_qpos_seq, object_pose_seq,
                                                                            reference_object_name, init_object_lift)
        self.hind_sight_environment_model(retarget_qpos_seq, object_pose_seq, self.object_names)

        retarget_qpos_seq, retarget_qvel_seq, retarget_qacc_seq, object_pose_seq = min_jerk_interpolate_replay_sequence(
            retarget_qpos_seq, object_pose_seq, action_time_step, collect_time_step)
        # retarget_qpos_seq, object_pose_seq = interpolate_replay_sequence(retarget_qpos_seq, object_pose_seq,
        #                                                                  action_time_step, collect_time_step)

        result = {}
        imitation_data = []
        num_samples = len(retarget_qpos_seq)
        result["model_data"] = [self.dump_mujoco_model()]

        for i in range(num_samples):
            qpos = self.filter.next(retarget_qpos_seq[i][:])
            self.sim.data.qpos[:qpos.shape[0]] = qpos
            self.sim.data.qvel[:qpos.shape[0]] = np.clip(retarget_qvel_seq[i][:], -3, 3)
            self.sim.data.qacc[:qpos.shape[0]] = np.clip(retarget_qacc_seq[i][:], -10, 10)
            for k, object_name in enumerate(self.object_names[1:3]):
                object_pose = object_pose_seq[i][object_name]
                self.sim.data.qpos[self.object_trans_qpos_indices[k]] = object_pose[:3, 3]
                self.sim.data.qpos[self.object_rot_qpos_indices[k]] = transforms3d.quaternions.mat2quat(
                    object_pose[:3, :3])

            imitation_data.append(self.fetch_imitation_data(self.act_mid, self.act_rng))
            if self.has_renderer:
                for _ in range(1):
                    self.render()
            self.sim.forward()

            # Break when the mug is located well for only 2 objects
            if len(self.object_names) < 3:
                obj_pos = self.data.body_xpos[self.object_bids[1]].ravel()
                target_pos = self.data.body_xpos[self.target_object_bids[1]].ravel()
                palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
                object_pos_to_palm = palm_pos - obj_pos
                target_pos_to_object = obj_pos - target_pos
                if np.linalg.norm(target_pos_to_object) < 0.003 and np.linalg.norm(object_pos_to_palm) > 0.15:
                    break

        # Pack entry together by key
        step_size = len(imitation_data)
        for key in ['observations', 'rewards', 'actions']:
            result[key] = np.stack([imitation_data[i][key] for i in range(step_size)], axis=0)
        result['sim_data'] = [imitation_data[i]["sim_data"] for i in range(step_size)]

        # Verbose
        duration = time.time() - tic
        print(f"Generating demo data {name} with {num_samples} samples takes {duration} seconds.")
        self.pack(self.init_sim_data)
        self.pack_mujoco_model(self.init_model_data)
        return result

    def hind_sight_environment_model(self, retarget_qpos_seq: List[np.ndarray],
                                     object_pose_seq: List[Dict[str, np.ndarray]], reference_object_names: List[str]):
        for reference_object_name in reference_object_names:
            for object_name, pose in object_pose_seq[-1].items():
                if object_name == reference_object_name:
                    target_body_pos = pose[:3, 3]
                    target_body_quat = transforms3d.quaternions.mat2quat(pose[:3, :3])
                    self.mjpy_model.body_pos[self.mjpy_model.body_name2id(f"{object_name}_target"),
                    0:3] = target_body_pos
                    self.mjpy_model.body_quat[self.mjpy_model.body_name2id(f"{object_name}_target"),
                    0:4] = target_body_quat

                    if object_name == "sugar_box":
                        self.mjpy_model.body_pos[self.mjpy_model.body_name2id(f"{object_name}_0"),
                        0:3] = target_body_pos
                        self.mjpy_model.body_quat[self.mjpy_model.body_name2id(f"{object_name}_0"),
                        0:4] = target_body_quat

    def strip_stacking(self, retarget_qpos_seq: List[np.ndarray], object_pose_seq: List[Dict[str, np.ndarray]]):
        # Not only strip empty value, but also strip position where the robot arm is behind origin
        retarget_qpos_seq, object_pose_seq = self.strip_negative_origin(retarget_qpos_seq, object_pose_seq)
        while len(object_pose_seq[0].keys()) < len(self.object_names):
            retarget_qpos_seq = retarget_qpos_seq[1:]
            object_pose_seq = object_pose_seq[1:]
        return retarget_qpos_seq, object_pose_seq


if __name__ == '__main__':
    from hand_imitation.misc.path_utils import get_project_root

    env = StackingDemonstration(True, object_names=["sugar_box", "mug", "large_clamp"], object_scales=[1, 1, 1])
    project_root = get_project_root()
    demo_path = "/home/sim/data/wanglab_dataset/dataset_v2/retargeting/stacking/all/seq_14/multi_post_demonstration_v3.pkl"
    # demo_path = "/home/sim/data/wanglab_dataset/tomato_soup_can_with_actions.pkl"
    demo = np.load(demo_path, allow_pickle=True)

    from hand_imitation.misc.joint_utils import filter_position_sequence
    import transforms3d
    from hand_imitation.env.utils.gif_recording import RecordingGif

    all_joints = np.stack([d["qpos"][:] for d in demo["sim_data"]], axis=0)
    hand_joints = all_joints[:, :30]
    object_position1 = all_joints[:, 30:33]
    object_quat1 = all_joints[:, 33:37]
    object_lie1 = []
    object_position2 = all_joints[:, 37:40]
    object_quat2 = all_joints[:, 40:44]
    object_lie2 = []
    for i in range(object_quat1.shape[0]):
        axis, angle = transforms3d.quaternions.quat2axangle(object_quat1[i])
        object_lie1.append(axis * angle)
    object_lie1 = np.stack(object_lie1, axis=0)
    for i in range(object_quat2.shape[0]):
        axis, angle = transforms3d.quaternions.quat2axangle(object_quat2[i])
        object_lie2.append(axis * angle)
    object_lie2 = np.stack(object_lie2, axis=0)

    base = 30
    hand_joints_filter = filter_position_sequence(hand_joints, wn=5, fs=base * 10)
    object_position_filter1 = filter_position_sequence(object_position1, wn=5, fs=base * 10)
    object_lie_filter1 = filter_position_sequence(object_lie1, wn=1, fs=base * 100)
    object_quat_filter1 = []
    object_position_filter2 = filter_position_sequence(object_position2, wn=5, fs=base * 10)
    object_lie_filter2 = filter_position_sequence(object_lie2, wn=1, fs=base * 100)
    object_quat_filter2 = []
    for i in range(object_quat1.shape[0]):
        angle = np.linalg.norm(object_lie_filter1[i])
        axis = object_lie1[i] / (angle + 1e-6)
        object_quat_filter1.append(transforms3d.quaternions.axangle2quat(axis, angle))
    object_quat_filter1 = np.stack(object_quat_filter1, axis=0)
    for i in range(object_quat2.shape[0]):
        angle = np.linalg.norm(object_lie_filter2[i])
        axis = object_lie2[i] / (angle + 1e-6)
        object_quat_filter2.append(transforms3d.quaternions.axangle2quat(axis, angle))
    object_quat_filter2 = np.stack(object_quat_filter2, axis=0)

    for i in range(len(demo["sim_data"])):
        demo["sim_data"][i]["qpos"][:30] = hand_joints_filter[i]
        demo["sim_data"][i]["qpos"][30:33] = object_position_filter1[i]
        demo["sim_data"][i]["qpos"][33:37] = object_quat_filter1[i]
        demo["sim_data"][i]["qpos"][37:40] = object_position_filter2[i]
        # demo["sim_data"][i]["qpos"][40:44] = object_quat_filter2[i]

    # env.replay_state(demo)
    with RecordingGif(env, "/home/sim/data/wanglab_dataset/dataset_v2/temp_video", camera_names=["frontview"],
                      format="mp4", freq=20) as video:
        env.replay_state(demo)
