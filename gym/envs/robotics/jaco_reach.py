# Usage
# python -m baselines.run --alg=her --env=JacoReach-v1 --num_timesteps=300000 --save_path=./baselines/policies/results
# python -m baselines.run --alg=her --env=JacoReach-v1 --num_timesteps=0 --load_path=./baselines/policies/results --play

import numpy as np
from gym import utils
from gym.envs.robotics import mujoco_env

from gym.envs.robotics.jaco import JacoEnv

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # print(goal_a.shape)
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class JacoReachEnv(JacoEnv):
    def __init__(self, with_rot=1,reward_type='sparse',distance_threshold=0.01):
        super().__init__(with_rot=with_rot)
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
        self.distance_threshold = distance_threshold
        self.reward_types = reward_type   # different from self.reward_type (Jaco had it originally and was not sure whether to remove it or not)
        self.reward_type += ["pick_reward", "success"]
        self.ob_type = self.ob_shape.keys()

        # mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        mujoco_env.MujocoEnv.__init__(self, "jaco_reach.xml", 4)
        utils.EzPickle.__init__(self)

    def set_norm(self, norm):
        self._norm = norm

    def set_context(self, context):
        self._context = context

    def _step(self, a):
        # if np.abs(a.any()) > 1:
            # print("Action: ",a)
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        # dist_hand = self._get_distance_hand('box')
        dist_hand = self._get_distance_hand('target')
        # box_z = self._get_box_pos()[2] # z coordinate of box
        # in_air = box_z > 0.04 # diameter of ball
        # on_ground = box_z < 0.04
        # in_hand = dist_hand < 0.04
        # # pick
        # if in_air and in_hand:
        #     pick_reward = self._config["pick_reward"] * box_z
        #     self._pick_count += 1
        #
        # # fail
        # if on_ground and self._pick_count > 0:
        #     done = True

        # success
        # if self._pick_count == 50:
        #     success = True
        #     # done = True
        #     print('success')

        reward = ctrl_reward + pick_reward
        # info = {"ctrl_reward_sum": ctrl_reward,
        #         "pick_reward_sum": pick_reward,
        #         "success_sum": success}
        info = {"is_success": self._is_success(ob['achieved_goal'], self.goal)}
        # print("Info: ",info)
        reward = self.compute_reward(ob['achieved_goal'],self.goal,info)
        # print(ob['achieved_goal'].shape)

        return ob, reward, done, info

    def _is_success(self,achieved_goal,desired_goal):
        d = goal_distance(achieved_goal,desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self):
        
        # grip_pos = self._get_hand_pos()
        grip_pos = self.sim.data.get_site_xpos('jaco_hand')[:3]
        # a = self._get_pos('jaco_link_hand')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('jaco_hand') * dt
        gripper_state = self.data.qpos
        gripper_vel = self.data.qvel*dt
        gripper_acc = self.data.qacc
        # print("Qpos shape", self.sim.data.qacc.shape)
        # print("Gripvel: ", grip_velp)
        # object_pos = self._get_pos('box')
        # print('jaco_pick.py pos target')
        # print(object_pos)
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(3)
        object_rel_pos = object_pos - grip_pos
        achieved_goal = grip_pos.copy()

        ob = np.concatenate([grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, np.clip(gripper_vel, -30, 30), gripper_acc])
        # ob = np.concatenate([grip_pos, gripper_state, grip_velp, np.clip(gripper_vel, -30, 30), gripper_acc])
        # print("object shape", ob.shape) #67
        # print("Achieved: ",achieved_goal)
        
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

    def compute_reward(self,achieved_goal, goal, info):
        # print("Achieved: ", achieved_goal.shape)
        # print("Desired: ", goal.shape)
        # print()
        d = goal_distance(achieved_goal,goal)
        # print(d.shape)
        # print("Distance: ",d)
        if self.reward_types == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d # dense reward
        # return  -(d > self.distance_threshold).astype(np.float32)
    
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
        # print("Initboxpos", self._init_box_pos)
        qpos[9:12] = self._init_box_pos

        self.set_state(qpos, qvel)

        self._pick_count = 0

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        # print('QPOS QVEL')
        # print(qpos,qvel)
        # print()
        self.set_state(qpos, qvel)

        self.reset_box()

        return self._get_obs()