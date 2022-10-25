import numpy as np
from hand_imitation.env.environments.allegro_env.allegro_relocate_dapg import RelocateEnvV0


class PID:
    def __init__(self, p, i, d, time_step):
        self.p = p
        self.i = i
        self.d = d
        self.error_sum = 0
        self.last_error = np.array([0])
        self.time_step = time_step

    def update(self, error):
        if self.last_error.sum() < 1e-6:
            self.last_error = error
        self.error_sum += error * self.time_step
        p_term = -self.p * error
        d_term = - (error - self.last_error) / self.time_step * self.d
        i_term = -self.error_sum * self.i
        self.last_error = error
        return p_term + i_term + d_term


class AllegroHandControl:
    def __init__(self, has_renderer,
                 render_gpu_device_id=-1,
                 primitive_name="sphere",
                 simple_env=True):
        self.env = RelocateEnvV0(has_renderer, render_gpu_device_id, primitive_name=primitive_name, simple=simple_env)
        self.stage1_pid = PID(np.array([0.01, 0.01, 0.05]), 0.00000, 0.02, self.env.mjpy_model.opt.timestep)
        self.stage2_pid = PID(0.03, 0.00, 0.0002, self.env.mjpy_model.opt.timestep)
        self.stage3_pid = PID(np.array([0.01, 0.01, 0.05]), 0.00000, 0.02, self.env.mjpy_model.opt.timestep)

    def stage_one(self):
        obs = self.env.reset()
        palm_to_object = obs[22:25]

        while np.linalg.norm(palm_to_object) > 0.10:
            error = np.copy(palm_to_object)
            error[0:3] -= np.array([0.04, 0.02, 0.08])
            ctrl_plus = self.stage1_pid.update(error)
            ctrl = self.env.data.ctrl[:]
            ctrl[:3] += ctrl_plus
            action = (ctrl - self.env.act_mid) / self.env.act_rng
            obs, reward, done, _ = self.record_and_step(action)
            palm_to_object = obs[22:25]

    def stage_two(self):
        target_pos = np.array([0.0] + [1.0] * 3 + [0.99] * 4 + [1.2] * 5 + [0.8] + [1.0] * 2)
        ctrl = self.env.data.ctrl[:]

        for i in range(100):
            error = ctrl[6:] - target_pos
            ctrl_plus = self.stage2_pid.update(error)
            ctrl = self.env.data.ctrl[:]
            ctrl[6:] += ctrl_plus
            action = (ctrl - self.env.act_mid) / self.env.act_rng
            self.record_and_step(action)

    def stage_three(self):
        obs = self.env._get_observations()
        target_to_object = obs[-3:]

        while np.linalg.norm(target_to_object) > 0.02 and np.linalg.norm(obs[-9:-6]) < 0.1:
            error = np.copy(target_to_object)
            ctrl_plus = self.stage3_pid.update(error)
            ctrl = self.env.data.ctrl[:]
            ctrl[:3] += ctrl_plus
            print(target_to_object, ctrl_plus)
            action = (ctrl - self.env.act_mid) / self.env.act_rng
            obs, reward, done, _ = self.record_and_step(action)
            target_to_object = obs[-3:]

    def wait_for_steps(self, steps):
        for _ in range(steps):
            ctrl = self.env.data.ctrl[:]
            action = (ctrl - self.env.act_mid) / self.env.act_rng
            self.record_and_step(action)

    def record_and_step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.has_renderer:
            self.env.render()

        return obs, reward, done, info


if __name__ == '__main__':
    control_policy = AllegroHandControl(True, simple_env=False)
    control_policy.stage_one()
    control_policy.wait_for_steps(50)
    control_policy.stage_two()
    control_policy.wait_for_steps(50)
    control_policy.stage_three()
