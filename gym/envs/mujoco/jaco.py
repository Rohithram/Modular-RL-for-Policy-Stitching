import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class JacoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, with_rot=1):
        # config
        self._with_rot = with_rot
        self._config = {"ctrl_reward": 1e-4}

        # env info
        self.reward_type = ["ctrl_reward"]
        self.ob_shape = {"joint": [31]}
        if self._with_rot == 0:
            self.ob_shape["joint"] = [24]  # 4 for orientation, 3 for velocity

    def set_environment_config(self, config):
        for k, v in config.items():
            self._config[k] = v

    def _ctrl_reward(self, a):
        ctrl_reward = -self._config["ctrl_reward"] * np.square(a).sum()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.sim.data.qvel).mean()
        ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.sim.data.qacc).mean()
        return ctrl_reward

    # get absolute coordinate
    def _get_pos(self, name):
        # print('NAME:',name)
        geom_idx = np.where([key == name for key in self.sim.model.geom_names])
        # print('GEOM IDX')
        # print(geom_idx[0])
        # print()
        if len(geom_idx[0]) > 0:
            # print('GEOM IDX',self.sim.data.geom_xpos[geom_idx[0][0]])
            return self.sim.data.geom_xpos[geom_idx[0][0]]
        body_idx = np.where([key == name for key in self.sim.model.body_names])
        # print('AVAILABLE:',self.sim.model.body_names)
        # print('BODY IDX')
        # print(body_idx[0])
        # print()
        if len(body_idx[0]) > 0:
            # print('BODY IDX',self.sim.body_pos[body_idx[0][0]])
            return self.sim.body_pos[body_idx[0][0]]
        raise ValueError

    def _get_box_pos(self):
        # changed from  return self._get_pos('box')
        return self._get_pos('target')

    def _get_target_pos(self):
        return self._get_pos('target')

    def _get_hand_pos(self):
        hand_pos = np.mean([self._get_pos(name) for name in [
            'jaco_link_hand', 'jaco_link_finger_1',
            'jaco_link_finger_2', 'jaco_link_finger_3']], 0)
        return hand_pos

    def _get_distance(self, name1, name2):
        pos1 = self._get_pos(name1)
        pos2 = self._get_pos(name2)
        return np.linalg.norm(pos1 - pos2)

    def _get_distance_hand(self, name):
        pos = self._get_pos(name)
        hand_pos = self._get_hand_pos()
        return np.linalg.norm(pos - hand_pos)

    def viewer_setup(self):
        #self.viewer.cam.trackbodyid = 1
        self.viewer.cam.trackbodyid = -1
        #self.viewer.cam.distance = self.sim.stat.extent * 2
        self.viewer.cam.distance = 2
        #self.viewer.cam.azimuth = 260
        self.viewer.cam.azimuth = 170
        #self.viewer.cam.azimuth = 90
        self.viewer.cam.lookat[0] = 0.5
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.elevation = -20

    def _sample_goal(self):
        # print('GOAL\n\n')
        # if self.has_object:
        #     print('DONE\n\n')
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        #     goal += self.target_offset
        #     goal[2] = self.height_offset
        #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
        #         goal[2] += self.np_random.uniform(0, 0.45)
        # else:
        #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        # goal = np.random.randint(0,2,1)
        goal = np.random.randint(0,2,3)
        return goal.copy()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('jaco_joint_finger_1')
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) #+ self.sim.data.get_site_xpos('jaco_joint_finger_1')
        gripper_rotation = np.array([1., 0., 1., 0.])
        # self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        # self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # print(self.sim.data.get_body_xpos('jaco_link_finger_1'))
        # print(self.sim.data.get_joint_qpos('jaco_joint_finger_1'))
        # print(self.sim.data.get_joint_qvel('jaco_joint_finger_1'))
        # self.sim.data.set_mocap_pos('jaco_link_finger_1', gripper_target)
        # self.sim.data.set_mocap_quat('jaco_link_finger_1', gripper_rotation)
        
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        # self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        # if self.has_object:
        #     self.height_offset = self.sim.data.get_site_xpos('object0')[2]
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('jaco_joint_finger_1').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('target')[2]
