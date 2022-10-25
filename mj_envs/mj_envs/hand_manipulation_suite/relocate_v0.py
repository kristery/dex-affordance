import numpy as np
from gym import utils
from mj_envs import mujoco_env
from mujoco_py import MjViewer
import os

from tpi.core.config import cfg


class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(curr_dir, 'assets/DAPG_relocate.xml')
        assert not cfg.CUSTOM_OBJECT or not cfg.CUSTOM_FINGERS
        # Change the object
        if cfg.CUSTOM_OBJECT:
            xml_file = 'assets/DAPG_relocate_{}.xml'.format(cfg.CUSTOM_OBJECT_TYPE)
            xml_path = os.path.join(curr_dir, xml_file)
            assert os.path.exists(xml_path)
        # Change the fingers
        if cfg.CUSTOM_FINGERS:
            xml_file = 'assets/DAPG_relocate_fingers_{}.xml'.format(cfg.CUSTOM_FINGERS_MASK)
            xml_path = os.path.join(curr_dir, xml_file)
            assert os.path.exists(xml_path)
        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)

        # Change actuator sensitivity (following DAPG)
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:] = np.array([0, -1, 0])

        # Change the dynamics
        if cfg.CUSTOM_MASS:
            self.sim.model.body_mass[6:28] *= cfg.CUSTOM_MASS_MUL
        if cfg.CUSTOM_FRICT:
            self.sim.model.geom_friction[:] *= cfg.CUSTOM_FRICT_MUL

        # Change object mass and size
        if cfg.CUSTOM_OBJ_MASS:
            self.sim.model.body_mass[-1] *= cfg.CUSTOM_OBJ_MASS_MUL
        if cfg.CUSTOM_OBJ_SIZE:
            self.sim.model.geom_size[-1] *= cfg.CUSTOM_OBJ_SIZE_MUL

        # Color for different dynamics visualization
        #self.sim.model.geom_rgba[6:-1] = np.array([1.0, 0.0, 0.0, 1.0])
        #self.sim.model.geom_rgba[6:-1] = np.array([0.850, 0.325, 0.098, 1.0])
        #self.sim.model.geom_rgba[6:-1] = np.array([0.564, 0.564, 0.564, 1.0])

        #from IPython import embed; embed()
        #import sys; sys.exit(0)

        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id('Object')
        utils.EzPickle.__init__(self)
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])

        #from IPython import embed; embed()
        #import sys; sys.exit(0)

    def _step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        if cfg.DENSE_REWARD:
            reward = -0.1*np.linalg.norm(palm_pos-obj_pos)              # take hand to object
            if obj_pos[2] > 0.04:                                       # if object off the table
                reward += 1.0                                           # bonus for lifting the object
                reward += -0.5*np.linalg.norm(palm_pos-target_pos)      # make hand go to target
                reward += -0.5*np.linalg.norm(obj_pos-target_pos)       # make object go to target
        else:
            reward = 0

        if np.linalg.norm(obj_pos-target_pos) < 0.1:
            reward += 10.0                                          # bonus for object close to target
        if np.linalg.norm(obj_pos-target_pos) < 0.05:
            reward += 20.0                                          # bonus for object "very" close to target

        return ob, reward, False, {}

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos-obj_pos, palm_pos-target_pos, obj_pos-target_pos])
       
    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid,0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid,1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid,2] = self.np_random.uniform(low=0.15, high=0.35)
        # Teaser, 45 deg cam
        #self.model.body_pos[self.obj_bid, 0] = -0.2
        #self.model.body_pos[self.obj_bid, 1] = 0.0
        #self.model.site_pos[self.target_obj_sid, 0] = 0.2
        #self.model.site_pos[self.target_obj_sid, 1] = 0.2
        #self.model.site_pos[self.target_obj_sid, 2] = 0.35
        # Teaser, side cam
        #self.model.body_pos[self.obj_bid, 0] = 0.0
        #self.model.body_pos[self.obj_bid, 1] = 0.0
        #self.model.site_pos[self.target_obj_sid, 0] = 0.0
        #self.model.site_pos[self.target_obj_sid, 1] = 0.2
        #self.model.site_pos[self.target_obj_sid, 2] = 0.25
        # Object view
        #self.model.body_pos[self.obj_bid, 0] = -0.25
        #self.model.body_pos[self.obj_bid, 1] = 0.0
        #self.model.site_pos[self.target_obj_sid, 0] = 0.2
        #self.model.site_pos[self.target_obj_sid, 1] = 0.0
        #self.model.site_pos[self.target_obj_sid, 2] = 0.2
        self.sim.forward()
        return self._get_obs()

    def gs(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos,
            qpos=qp, qvel=qv)

    def ss(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 0
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            if np.sum(path['rewards']) > 500:  # ball close to target for at least 1/4th the trajectory
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    def replay(self, obs_list, interval=10):
        self.reset()
        for i, obs in enumerate(obs_list):
            self.sim.data.qpos[:-3] = obs[:33]
            target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
            obj_pos = obs[-3:] + target_pos
            self.sim.data.body_xpos[self.obj_bid] = obj_pos
            self.sim.step()
            for _ in range(interval):
                self.mj_render()
