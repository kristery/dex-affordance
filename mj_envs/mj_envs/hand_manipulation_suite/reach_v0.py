import numpy as np
import os

from gym import utils
from mj_envs import mujoco_env
from mujoco_py import MjViewer


class ReachEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        # Define the fields
        self.grasp_id = 0
        self.target_id = 0
        self.act_mid = None
        self.act_rng = None
        # Construct the mujoco env
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(cur_dir, 'assets/DAPG_reach.xml')
        mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip=5)
        # Adjust actuator sensitivity
        self.sim.model.actuator_gainprm[ \
            self.sim.model.actuator_name2id('A_WRJ1'): \
            self.sim.model.actuator_name2id('A_WRJ0') + 1, \
            : \
        ] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[ \
            self.sim.model.actuator_name2id('A_FFJ3'): \
            self.sim.model.actuator_name2id('A_THJ0') + 1, \
            : \
        ] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[ \
            self.sim.model.actuator_name2id('A_WRJ1'): \
            self.sim.model.actuator_name2id('A_WRJ0') + 1, \
            : \
        ] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[ \
            self.sim.model.actuator_name2id('A_FFJ3'): \
            self.sim.model.actuator_name2id('A_THJ0') + 1, \
            : \
        ] = np.array([0, -1, 0])
        # Required for mujoco envs
        utils.EzPickle.__init__(self)
        # Palm and raget id
        self.grasp_id = self.sim.model.site_name2id('S_grasp')
        self.target_id = self.sim.model.site_name2id('target')
        # Action mid and range
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def _step(self, a):
        # Perform the action
        a = np.clip(a, -1.0, 1.0)
        if self.act_mid is not None:
            a = self.act_mid + a * self.act_rng
        else:
            a = a
        self.do_simulation(a, self.frame_skip)
        # Retrieve the observation
        ob = self._get_obs()
        palm_pos = self.data.site_xpos[self.grasp_id].ravel()
        target_pos = self.data.site_xpos[self.target_id].ravel()
        # Compute the reward
        dist = np.linalg.norm(palm_pos - target_pos)
        reward = -0.1 * dist
        if dist < 0.1:
            reward += 10.0
        if dist < 0.05:
            reward += 20.0
        return ob, reward, False, {}

    def _get_obs(self):
        qp = self.data.qpos.ravel()
        palm_pos = self.data.site_xpos[self.grasp_id].ravel()
        target_pos = self.data.site_xpos[self.target_id].ravel()
        return np.concatenate([qp[:-6], palm_pos - target_pos])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.site_pos[self.target_id, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_id, 1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_id, 2] = self.np_random.uniform(low=0.15, high=0.35)
        self.sim.forward()
        return self._get_obs()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 0
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(paths['rewards']) > 500:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
