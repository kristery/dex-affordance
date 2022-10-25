import numpy as np
from transforms3d.euler import euler2quat

from hand_imitation.env.environments.base import MujocoEnv
from hand_imitation.env.models import TableArena
from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.mjcf_utils import xml_path_completion
from hand_imitation.env.utils.random import np_random

ADD_BONUS_REWARDS = True


class PenEnvV0(MujocoEnv):
    def __init__(self, has_renderer, render_gpu_device_id=-1):
        self.np_random = None
        self.seed()
        super().__init__(has_renderer=has_renderer, has_offscreen_renderer=False, render_camera=None,
                         render_gpu_device_id=render_gpu_device_id, control_freq=100, horizon=200, ignore_done=True,
                         hard_reset=False)

        # Setup action range
        self.act_mid = np.mean(self.mjpy_model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.mjpy_model.actuator_ctrlrange[:, 1] - self.mjpy_model.actuator_ctrlrange[:, 0])

    def _pre_action(self, action, policy_step=False):
        action = np.clip(action, -1.0, 1.0)
        action = self.act_mid + action * self.act_rng  # mean center and scale
        self.sim.data.ctrl[:] = action

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.set_state(self.sim_state_initial)
        self.sim.forward()

        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.mjpy_model.body_quat[self.target_obj_bid] = euler2quat(*desired_orien)

    def _get_observations(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orientation = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orientation = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[
            self.tar_b_sid]) / self.tar_length
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orientation, desired_orientation,
                               obj_pos - desired_pos, obj_orientation - desired_orientation])

    def reward(self, action):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orientation = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orientation = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[
            self.tar_b_sid]) / self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos - desired_loc)
        reward = -dist
        # orientation cost
        orientation_similarity = np.dot(obj_orientation, desired_orientation)
        reward += orientation_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orientation_similarity > 0.9:
                reward += 10
            if dist < 0.075 and orientation_similarity > 0.95:
                reward += 50

        # penalty for dropping the pen
        if obj_pos[2] < 0.075:
            reward -= 5

        return reward

    def _setup_references(self):
        self.sim.forward()
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.pen_length = np.linalg.norm(
            self.sim.data.site_xpos[self.obj_t_sid] - self.sim.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(
            self.sim.data.site_xpos[self.tar_t_sid] - self.sim.data.site_xpos[self.tar_b_sid])

    def _load_model(self):
        arena = TableArena(table_full_size=(0.9, 0.9, 0.05), table_friction=(1, 0.5, 0.01), table_offset=(0, 0, 1.0),
                           bottom_pos=(0, 0, -1), has_legs=True)
        xml_file = xml_path_completion("adroit/adroit_dapg_pen.xml")
        robot = MujocoXML(xml_file)
        robot.merge(arena, merge_body="default")

        self.model = robot
        self.model.save_model("dapg_pen_temp.xml")

    @property
    def action_spec(self):
        high = np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 1])
        low = -1.0 * np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 0])
        return low, high

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        desired_orien = self.mjpy_model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.mjpy_model.body_quat[self.target_obj_bid] = desired_orien
        self.sim.forward()

    def set_state(self, qpos, qvel):
        import mujoco_py
        assert qpos.shape == (self.mjpy_model.nq,) and qvel.shape == (self.mjpy_model.nv,)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()
