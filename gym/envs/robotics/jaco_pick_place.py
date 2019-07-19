import numpy as np

from gym import utils
from gym.envs.robotics import mujoco_env

from gym.envs.robotics.jaco import JacoEnv

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # print(goal_a.shape)
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class JacoPickAndPlaceEnv(JacoEnv):
    def __init__(self, with_rot=1, reward_type='sparse',distance_threshold=0.08):
        super().__init__(with_rot=with_rot)

        # config
        self._config.update({
            "pick_reward": 100,
            "hold_reward": 2,
            "guide_reward": 2,
            "success_reward": 1,
            "random_box": 1,
            "init_randomness": 0.01,
            "random_steps": 10,
            "hold_duration": 50,
        })

        # state
        self._t = 0
        self._hold_duration = 0
        self._picked = False
        self._pick_height = 0
        self._dist_box = 0
        self.target_in_the_air = False

        # env info
        self.reward_types = reward_type
        self.reward_type += ["guide_reward", "pick_reward", "hold_reward",
                             "success_reward", "success"]
        self.ob_type = self.ob_shape.keys()
        self.distance_threshold = distance_threshold
        mujoco_env.MujocoEnv.__init__(self, "jaco_pick_place.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self._t += 1
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        guide_reward = 0
        pick_reward = 0
        hold_reward = 0
        ctrl_reward = self._ctrl_reward(a)
        success_reward = 0

        hand_pos = self._get_hand_pos()
        box_z = self._get_box_pos()[2]
        dist_box = self._get_distance_hand('target')
        in_hand = dist_box < 0.06
        in_air = box_z > 0.05

        if in_hand and in_air:
            self._picked = True

            # pick up
            if self._pick_height < min(self._target_pos[2], box_z):
                pick_reward = self._config["pick_reward"] * \
                    (min(self._target_pos[2], box_z) - self._pick_height)
                self._pick_height = box_z

            # hold
            dist = np.linalg.norm(self._target_pos - self._get_box_pos())
            hold_reward = self._config["hold_reward"] * (1 - dist)
            self._hold_duration += 1

            # success
            if self._config['hold_duration'] == self._hold_duration:
                print('success pick!', self._get_box_pos())
                done = success = True
                success_reward = self._config["success_reward"] * (200 - self._t)

        # guide hand to the box
        if not self._picked:
            guide_reward = self._config["guide_reward"] * (self._dist_box - dist_box)
            self._dist_box = dist_box

        # if self._picked and not in_hand:
            # done = True

        # reward = ctrl_reward + pick_reward + hold_reward + guide_reward + success_reward
        # info = {"ctrl_reward": ctrl_reward,
        #         "pick_reward": pick_reward,
        #         "hold_reward": hold_reward,
        #         "guide_reward": guide_reward,
        #         "success_reward": success_reward,
        #         "success": success}
        info = {"is_success": self._is_success(ob['achieved_goal'], self.goal)}
        reward = self.compute_reward(ob['achieved_goal'], self.goal, info)
        return ob, reward, done, info

    def _is_success(self,achieved_goal,desired_goal):
        d = goal_distance(achieved_goal,desired_goal)
        # print("Distance: ",d)
        # print(d < self.distance_threshold)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self):
        # grip_pos = self._get_hand_pos()
        grip_pos = self.sim.data.get_site_xpos('jaco_hand')
        # a = self._get_pos('jaco_link_hand')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('jaco_hand') * dt
        gripper_state = self.data.qpos
        gripper_vel = self.data.qvel*dt
        gripper_acc = self.data.qacc
        # print("Qpos shape", self.sim.data.qacc.shape)
        # print("Gripvel: ", grip_velp)
        # object_pos = self.sim.data.get_site_xpos('box')
        # object_pos = self._get_target_pos()
        # rotations
        # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('box'))
        # velocities
        # object_velp = self.sim.data.get_site_xvelp('box') * dt
        # object_velr = self.sim.data.get_site_xvelr('box') * dt
        # gripper state
        object_pos = self.sim.data.get_site_xpos('object')
        # rotations
        # object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('box'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object') * dt
        object_velr = self.sim.data.get_site_xvelr('object') * dt
        object_rel_pos = object_pos - grip_pos
        # object_velp -= grip_velp
        # print('jaco_pick.py pos target')
        # print(object_pos)
        # target_pos = self._get_pos('box')
        # object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(3)
        # object_rel_pos = object_pos - grip_pos
        # achieved_goal = grip_pos.copy()
        achieved_goal = np.squeeze(object_pos.copy())

        ob = np.concatenate([grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, 
            object_velp.ravel(), object_velr.ravel(), grip_velp, np.clip(gripper_vel, -30, 30), gripper_acc])
        # print("object shape", ob.shape) #67

        return {
            'observation': ob.copy(),
            'achieved_goal': achieved_goal.copy(),
             'desired_goal': (self.goal.copy()),
        }

    def get_ob_dict(self, ob):
        if len(ob.shape) > 1:
            return {
                'joint': ob[:, :31],
                'acc': ob[:, 31:46],
                'hand': ob[:, 46:49]
            }
        else:
            return {
                'joint': ob[:31],
                'acc': ob[31:46],
                'hand': ob[46:49]
            }

    def compute_reward(self,achieved_goal,goal,info):
        d = goal_distance(achieved_goal,goal)
        # print(d.shape)
        # print(np.mean(d))
        if self.reward_types == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d # dense reward

    def reset_box(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()

        # set box's initial pose
        self._init_box_pos = np.asarray(
            [0.5 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.2 + np.random.uniform(0, 0.1) * self._config["random_box"],
             0.03])
        qpos[9:12] = self._init_box_pos
        init_randomness = self._config["init_randomness"]
        qpos[12:16] = self.init_qpos[12:16] + np.random.uniform(low=-init_randomness,
                                                                high=init_randomness,
                                                                size=4)
        qvel[9:15] = self.init_qvel[9:15] + np.random.uniform(low=-init_randomness,
                                                              high=init_randomness,
                                                              size=6)
        self.set_state(qpos, qvel)

        self._t = 0
        self._hold_duration = 0
        self._pick_height = 0
        self._picked = False
        self._dist_box = np.linalg.norm(self._get_hand_pos() - self._init_box_pos)
        self._target_pos = self._init_box_pos.copy()
        self._target_pos[2] = 0.3

    def reset_model(self):
        init_randomness = self._config["init_randomness"]
        qpos = self.init_qpos + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-init_randomness,
                                                  high=init_randomness,
                                                  size=self.model.nv)
        self.set_state(qpos, qvel)

        self.reset_box()

        # more perturb
        for _ in range(int(self._config["random_steps"])):
            self.do_simulation(self.unwrapped.action_space.sample(), self.frame_skip)

        return self._get_obs()

    def is_terminate(self, ob, success_length=50, init=False, env=None):
        if init:
            self.count_evaluate = 0
            self.success = True

        box_pos = ob[9:12]
        hand_pos = ob[46:49]
        dist_box = np.linalg.norm(box_pos - hand_pos)
        box_z = box_pos[2]
        in_hand = dist_box < 0.06
        in_air = box_z > 0.05

        if not in_hand and self.count_evaluate > 0:
            self.success = False

        if in_air and in_hand:
            self.count_evaluate += 1

        return self.success and self.count_evaluate >= success_length

