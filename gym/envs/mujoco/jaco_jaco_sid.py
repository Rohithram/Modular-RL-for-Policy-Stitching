import numpy as np
from gym import utils as utils_gym # to not get confused with gym.envs.robotics utils
from gym.envs.robotics import rotations, robot_env, utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import gym
import os

# for attr in dir(mujoco_env):
#     print("obj.%s = %r" % (attr, getattr(mujoco_env, attr)))
#
# print('\n\n')
# print(.__dict__.keys())
# print('\n\n')

# super(JacoEnv, self).__init__(model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# super(mujoco_env.MujocoEnv).__init__(model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# super(JacoEnv,self).__init__(model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# super(mujoco_env.MujocoEnv).__init__(self,model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# super(utils.EzPickle).__init__(self)
# super().__init__()
# super()
# JacoEnv.__init__(model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# mujoco_env.MujocoEnv.__init__(self,model_path=model_path, frame_skip=4, rgb_rendering_tracking=True)
# utils.EzPickle.__init__(self)
class JacoEnv(mujoco_env.MujocoEnv, utils_gym.EzPickle):
    """Superclass for all Jaco environments.
    """
    def __init__(self, model_path, frame_skip, target_offset,
            obj_range, target_range, distance_threshold,initial_qpos,target_in_the_air,
            gripper_extra_height,block_gripper, has_object=True, with_rot=1):
        self._with_rot = with_rot
        self._config = {"ctrl_reward": 1e-4}

        # env info
        self.reward_type = ["ctrl_reward"]
        self.ob_shape = {"joint": [31]}
        if self._with_rot == 0:
            self.ob_shape["joint"] = [24]  # 4 for orientation, 3 for velocity
        
        self.has_object = has_object
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.target_in_the_air = target_in_the_air

        # for picking
        self._config.update({
            "pick_reward": 10,
            "random_box": 0.1,
        })
        self._context = None
        self._norm = False

        # state
        self._pick_count = 0
        self._init_box_pos = np.asarray([0.5, 0.0, 0.04])

        # env info
        self.reward_type += ["pick_reward", "success"]
        self.ob_type = self.ob_shape.keys()
        # model_path, frame_skip,n_actions,init_qpos,rgb_rendering_tracking=True
        super(JacoEnv,self).__init__(model_path=model_path, initial_qpos = initial_qpos,frame_skip=4,n_actions=4, rgb_rendering_tracking=True)
        utils_gym.EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    ## From FetchEnv
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _render_callback(self):
        print('CALLBACK\n\n')
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _step_callback(self):
        print('STEPCALLBACK\n\n')
        if self.block_gripper:
            self.sim.data.set_joint_qpos('robot0:l_gripper_finger_joint', 0.)
            self.sim.data.set_joint_qpos('robot0:r_gripper_finger_joint', 0.)
            self.sim.forward()

    def _reset_sim(self):
        print('RESET SIM\n\n')
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        print('GOAL\n\n')
        if self.has_object:
            print('DONE\n\n')
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        # goal = np.array([1,1])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        print('IS SUCCESS\n\n')
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

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
        print('DONE Setting UP\n')

    def render(self, mode='human', width=500, height=500):
        return super(JacoEnv, self).render(mode, width, height)

    ## From Changed Jaco
    # ----------------------------
    def _get_obs(self):
        print('GET OBS\n\n')
        # positions
        grip_pos = self.sim.data.get_joint_qpos('jaco_joint_finger_1')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_joint_qvel('jaco_joint_finger_1') * dt
        object_pos = self._get_pos('target')
        object_rel_pos = object_pos - grip_pos
        achieved_goal = grip_pos.copy()
        print('Position')
        print(grip_pos,object_pos,object_rel_pos)
        print()
        obs = np.concatenate([
            [grip_pos], object_pos.ravel(), object_rel_pos.ravel()])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('jaco_joint_finger_1')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

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
        print('GET POS\n\n')
        print("List of parts {}\n".format(self.sim.model.geom_names))
        geom_idx = np.where([key == name for key in self.sim.model.geom_names])
        if len(geom_idx[0]) > 0:
            return self.sim.data.geom_xpos[geom_idx[0][0]]
        body_idx = np.where([key == name for key in self.sim.model.body_names])
        if len(body_idx[0]) > 0:
            return self.sim.body_pos[body_idx[0][0]]
        raise ValueError

    def _get_box_pos(self):
        return self._get_pos('box')

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

    def set_norm(self, norm):
        self._norm = norm

    def set_context(self, context):
        self._context = context

    def get_ob_dict(self, ob):
        return {'joint': ob}

    def reset_box(self):
        qpos = self.sim.data.qpos.ravel().copy()
        qvel = self.sim.data.qvel.ravel().copy()

        # set box's initial position
        sx, sy, ex, ey = -1, -1, 1, 1
        if self._context == 0:
            sx, sy = 0, 0
        elif self._context == 1:
            ex, sy = 0, 0
        elif self._context == 2:
            sx, ey = 0, 0
        elif self._context == 3:
            ex, ey = 0, 0

        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(sx, ex) * self._config["random_box"],
             0.1 + np.random.uniform(sy, ey) * self._config["random_box"],
             0.04])
        qpos[9:12] = self._init_box_pos

        self.set_state(qpos, qvel)

        self._pick_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.reset_box()
        return self._get_obs()

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        dist_hand = self._get_distance_hand('box')
        box_z = self._get_box_pos()[2]
        in_air = box_z > 0.08
        on_ground = box_z < 0.08
        in_hand = dist_hand < 0.08

        # pick
        if in_air and in_hand:
            pick_reward = self._config["pick_reward"] * box_z
            self._pick_count += 1

        # fail
        if on_ground and self._pick_count > 0:
            done = True

        # success
        if self._pick_count == 50:
            success = True
            done = True
            print('success')

        reward = ctrl_reward + pick_reward
        info = {"ctrl_reward_sum": ctrl_reward,
                "pick_reward_sum": pick_reward,
                "success_sum": success}
        return ob, reward, done, info

    # def step(self, action):
    #     "From robot_env"
    #     action = np.clip(action, self.action_space.low, self.action_space.high)
    #     self._set_action(action)
    #     self.sim.step()
    #     self._step_callback()
    #     obs = self._get_obs()

    #     done = False
    #     info = {
    #         'is_success': self._is_success(obs['achieved_goal'], self.goal),
    #     }
    #     reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
    #     return obs, reward, done, info

##### ORIG

# import numpy as np
# from gym import utils
# from gym.envs.mujoco import mujoco_env
#
#
# class JacoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
#     '''Superclass for all Jaco environments
#     '''
#     def __init__(self, with_rot=1):
#         # config
#         self._with_rot = with_rot
#         self._config = {"ctrl_reward": 1e-4}
#
#         # env info
#         self.reward_type = ["ctrl_reward"]
#         self.ob_shape = {"joint": [31]}
#         if self._with_rot == 0:
#             self.ob_shape["joint"] = [24]  # 4 for orientation, 3 for velocity
#
#     def set_environment_config(self, config):
#         for k, v in config.items():
#             self._config[k] = v
#
#     def _ctrl_reward(self, a):
#         ctrl_reward = -self._config["ctrl_reward"] * np.square(a).sum()
#         ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.sim.data.qvel).mean()
#         ctrl_reward += -self._config["ctrl_reward"] ** 2 * np.abs(self.sim.data.qacc).mean()
#         return ctrl_reward
#
#     # get absolute coordinate
#     def _get_pos(self, name):
#         # print(self.sim.model.geom_names)
#         geom_idx = np.where([key == name for key in self.sim.model.geom_names])
#         if len(geom_idx[0]) > 0:
#             return self.sim.data.geom_xpos[geom_idx[0][0]]
#         body_idx = np.where([key == name for key in self.sim.model.body_names])
#         if len(body_idx[0]) > 0:
#             return self.sim.body_pos[body_idx[0][0]]
#         raise ValueError
#
#     def _get_box_pos(self):
#         return self._get_pos('box')
#
#     def _get_target_pos(self):
#         return self._get_pos('target')
#
#     def _get_hand_pos(self):
#         hand_pos = np.mean([self._get_pos(name) for name in [
#             'jaco_link_hand', 'jaco_link_finger_1',
#             'jaco_link_finger_2', 'jaco_link_finger_3']], 0)
#         return hand_pos
#
#     def _get_distance(self, name1, name2):
#         pos1 = self._get_pos(name1)
#         pos2 = self._get_pos(name2)
#         return np.linalg.norm(pos1 - pos2)
#
#     def _get_distance_hand(self, name):
#         pos = self._get_pos(name)
#         hand_pos = self._get_hand_pos()
#         return np.linalg.norm(pos - hand_pos)
#
#     def viewer_setup(self):
#         #self.viewer.cam.trackbodyid = 1
#         self.viewer.cam.trackbodyid = -1
#         #self.viewer.cam.distance = self.sim.stat.extent * 2
#         self.viewer.cam.distance = 2
#         #self.viewer.cam.azimuth = 260
#         self.viewer.cam.azimuth = 170
#         #self.viewer.cam.azimuth = 90
#         self.viewer.cam.lookat[0] = 0.5
#         self.viewer.cam.lookat[1] = 0
#         self.viewer.cam.lookat[2] = 0.5
#         self.viewer.cam.elevation = -20
#
#     def set_norm(self, norm):
#         self._norm = norm
#
#     def set_context(self, context):
#         self._context = context
#
#     def _step(self, a):
#         self.do_simulation(a, self.frame_skip)
#         ob = self._get_obs()
#         done = False
#         success = False
#         pick_reward = 0
#         ctrl_reward = self._ctrl_reward(a)
#
#         dist_hand = self._get_distance_hand('box')
#         box_z = self._get_box_pos()[2]
#         in_air = box_z > 0.08
#         on_ground = box_z < 0.08
#         in_hand = dist_hand < 0.08
#
#         # pick
#         if in_air and in_hand:
#             pick_reward = self._config["pick_reward"] * box_z
#             self._pick_count += 1
#
#         # fail
#         if on_ground and self._pick_count > 0:
#             done = True
#
#         # success
#         if self._pick_count == 50:
#             success = True
#             done = True
#             print('success')
#
#         reward = ctrl_reward + pick_reward
#         info = {"ctrl_reward_sum": ctrl_reward,
#                 "pick_reward_sum": pick_reward,
#                 "success_sum": success}
#         return ob, reward, done, info
#
#     def _get_obs(self):
#         qpos = self.sim.data.qpos
#         qvel = self.sim.data.qvel
#         finger = self.sim.data.get_joint_qpos("jaco_joint_finger_3")
#         finger_vel = self.sim.data.get_joint_qvel("jaco_joint_finger_3")
#         print(qpos)
#         print(qvel)
#         # print('HAND')
#         # print(self._get_hand_pos())
#         # print('BOX')
#         # print(self._get_box_pos())
#         # print('TARGET')
#         # print(self._get_target_pos())
#         # print('Hand dist')
#         # print(self._get_distance('jaco_link_finger_1','jaco_link_finger_2'))
#         # print()
#         obs = np.concatenate([qpos, qvel]).ravel()
#         if self._norm:
#             std = [5, 10, 20, 50, 100, 150, 0.2, 0.2,
#                    0.2, 1, 0.2, 1, 1, 0.2, 0.2, 0.2,
#                    50, 50, 70, 70, 100, 100, 50, 50,
#                    50, 5, 2, 10, 50, 50, 50]
#             obs /= std
#         return obs
#
#     def get_ob_dict(self, ob):
#         return {'joint': ob}
#
#     def reset_box(self):
#         qpos = self.sim.data.qpos.ravel().copy()
#         qvel = self.sim.data.qvel.ravel().copy()
#
#         # set box's initial position
#         sx, sy, ex, ey = -1, -1, 1, 1
#         if self._context == 0:
#             sx, sy = 0, 0
#         elif self._context == 1:
#             ex, sy = 0, 0
#         elif self._context == 2:
#             sx, ey = 0, 0
#         elif self._context == 3:
#             ex, ey = 0, 0
#
#         self._init_box_pos = np.asarray(
#             [0.5 + np.random.uniform(sx, ex) * self._config["random_box"],
#              0.1 + np.random.uniform(sy, ey) * self._config["random_box"],
#              0.04])
#         qpos[9:12] = self._init_box_pos
#
#         self.set_state(qpos, qvel)
#
#         self._pick_count = 0
#
#     def reset_model(self):
#         qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
#         qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
#         self.set_state(qpos, qvel)
#
#         self.reset_box()
#
#         return self._get_obs()
#
#
# # add position and sample_goal
#
#     ###### RENDER TO BE ADDED ######
#
# #### add goal function  here
