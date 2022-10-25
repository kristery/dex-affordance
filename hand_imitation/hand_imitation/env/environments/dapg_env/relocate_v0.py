from xml.etree import ElementTree as ET

import numpy as np

from hand_imitation.env.environments.base import MujocoEnv
from hand_imitation.env.models import TableArena
from hand_imitation.env.models.base import MujocoXML
from hand_imitation.env.utils.mjcf_utils import xml_path_completion, find_elements
from hand_imitation.env.utils.random import np_random

ADD_BONUS_REWARDS = True


class RelocateEnvV0(MujocoEnv):
    def __init__(self, has_renderer, render_gpu_device_id=-1, primitive_name="sphere",
                 primitive_size=(0.035, 0.035, 0.035)):
        self.np_random = None
        self.primitive_name = primitive_name
        self.primitive_size = primitive_size
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
        self.sim.forward()
        self.sim.set_state(self.sim_state_initial)
        self.sim.forward()
        self.data.qpos[-7] = self.np_random.uniform(low=-0.15, high=0.15)
        self.data.qpos[-6] = self.np_random.uniform(low=-0.15, high=0.3)
        self.mjpy_model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.mjpy_model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.mjpy_model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(low=0.15, high=0.35)

    def _get_observations(self):
        qp = self.data.qpos.ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:30], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])

    def reward(self, action):
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
        if obj_pos[2] > 0.04:  # if object off the table
            reward += 1.0  # bonus for lifting the object
            reward += -0.5 * np.linalg.norm(palm_pos - target_pos)  # make hand go to target
            reward += -0.5 * np.linalg.norm(obj_pos - target_pos)  # make object go to target

        if ADD_BONUS_REWARDS:
            if np.linalg.norm(obj_pos - target_pos) < 0.1:
                reward += 10.0  # bonus for object close to target
            if np.linalg.norm(obj_pos - target_pos) < 0.05:
                reward += 20.0  # bonus for object "very" close to target

        return reward

    def _setup_references(self):
        self.target_obj_sid = self.sim.model.site_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id('S_grasp')
        self.obj_bid = self.sim.model.body_name2id(self.object_body_name)

    def _load_model(self):
        arena = TableArena(table_full_size=(0.9, 0.9, 0.05), table_friction=(1, 0.5, 0.01), table_offset=(0, 0, 1.0),
                           bottom_pos=(0, 0, -1), has_legs=True)
        xml_file = xml_path_completion("adroit/adroit_dapg_relocate.xml")
        robot = MujocoXML(xml_file)

        # Add target object position for relocation object
        arena.add_primitive_object(primitive_name=self.primitive_name, primitive_size=self.primitive_size,
                                   pos=[0, 0, self.primitive_size[2]], quat=[1, 0, 0, 0],
                                   friction="1 0.5 0.01", margin="0.0005")
        self.object_body_name = arena.objects[0].body_name
        object_target = ET.Element("site", name="target", pos="-0.0007 0.0 0.2", size="0.04", rgba="0 0 1 0.125")
        arena.worldbody.append(object_target)
        robot.merge(arena, merge_body="default")

        # Modify object dynamics property to match DAPG
        object_body = find_elements(root=robot.worldbody, tags="body", attribs={"name": self.object_body_name},
                                    return_first=True)
        object_geom = find_elements(root=object_body, tags="geom", return_first=True)
        object_geom.set("condim", "4")
        object_joint = find_elements(root=object_body, tags="joint", return_first=True)
        object_joint.set("damping", "0")
        object_joint.set("armature", "0.001")
        object_joint.set("frictionloss", "0.001")

        self.model = robot
        # self.model.save_model("dapg_relocation_temp.xml")

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    @property
    def action_spec(self):
        high = np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 1])
        low = -1.0 * np.ones_like(self.mjpy_model.actuator_ctrlrange[:, 0])
        return low, high

    def set_state(self, qpos, qvel):
        import mujoco_py
        assert qpos.shape == (self.mjpy_model.nq,) and qvel.shape == (self.mjpy_model.nv,)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.sim.data.body_xpos[self.obj_bid].ravel()
        palm_pos = self.sim.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.sim.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos, qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.set_state(qp, qv)
        self.mjpy_model.body_pos[self.obj_bid] = obj_pos
        self.mjpy_model.site_pos[self.target_obj_sid] = target_pos
        self.sim.forward()

    @property
    def spec(self):
        this_spec = Spec(self._get_observations().shape[0], self.action_spec[0].shape[0])
        return this_spec

    def set_seed(self, seed=None):
        return self.seed(seed)


class Spec:
    def __init__(self, observation_dim, action_dim):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
