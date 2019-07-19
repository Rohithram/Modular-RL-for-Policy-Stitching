import numpy as np
from gym import utils
from gym.envs.robotics import mujoco_env

from gym.envs.robotics.jaco import JacoEnv

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # print(goal_a.shape)
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class JacoPickEnv(JacoEnv):
    def __init__(self, with_rot=1,reward_type='sparse',distance_threshold=0.08):
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

        mujoco_env.MujocoEnv.__init__(self, "jaco_pick.xml", 4)
        utils.EzPickle.__init__(self)

    def set_norm(self, norm):
        self._norm = norm

    def set_context(self, context):
        self._context = context

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        done = False
        success = False
        pick_reward = 0
        ctrl_reward = self._ctrl_reward(a)

        # dist_hand = self._get_distance_hand('box')
        dist_hand = self._get_distance_hand('target')
        box_z = self._get_box_pos()[2] # z coordinate of box
        in_air = box_z > 0.04 # diameter of ball
        on_ground = box_z < 0.04
        in_hand = dist_hand < 0.04
        # # pick
        # if in_air and in_hand:
        #     pick_reward = self._config["pick_reward"] * box_z
        #     self._pick_count += 1


        # if in_hand and in_air:
        #     self._picked = True

        #     # pick up
        #     if self._pick_height < min(self._target_pos[2], box_z):
        #         pick_reward = self._config["pick_reward"] * \
        #             (min(self._target_pos[2], box_z) - self._pick_height)
        #         self._pick_height = box_z

        #     # hold
        #     dist = np.linalg.norm(self._target_pos - self._get_box_pos())
        #     hold_reward = self._config["hold_reward"] * (1 - dist)
        #     self._hold_duration += 1

        #     # success
        #     if self._config['hold_duration'] == self._hold_duration:
        #         print('success pick!', self._get_box_pos())
        #         done = success = True
        #         success_reward = self._config["success_reward"] * (200 - self._t)

        # # guide hand to the box
        # if not self._picked:
        #     guide_reward = self._config["guide_reward"] * (self._dist_box - dist_box)
        #     self._dist_box = dist_box

        # if self._picked and not in_hand:
        #     done = True

        # reward = ctrl_reward + pick_reward + hold_reward + guide_reward + success_reward
        info = {"is_success": self._is_success(ob['achieved_goal'], self.goal)}
        # print("Info: ",info)
        reward = self.compute_reward(ob['achieved_goal'],self.goal,info)
        return ob, reward, done, info

    def _is_success(self,achieved_goal,desired_goal):
        d = goal_distance(achieved_goal,desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _get_obs(self):
        
        grip_pos = self.sim.data.get_site_xpos('jaco_hand')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('jaco_hand') * dt
        gripper_state = self.data.qpos
        gripper_vel = self.data.qvel*dt
        gripper_acc = self.data.qacc
        # print("Qpos shape", self.sim.data.qacc.shape)
        # print("Gripvel: ", grip_velp)
        # object_pos = self._get_pos('ball')
        # print('jaco_pick.py pos target')
        # print(object_pos)
        # target_pos = self._get_pos('box')
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(3)
        object_rel_pos = object_pos - grip_pos
        achieved_goal = grip_pos.copy()

        ob = np.concatenate([grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state,
            grip_velp, np.clip(gripper_vel, -30, 30), gripper_acc])
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