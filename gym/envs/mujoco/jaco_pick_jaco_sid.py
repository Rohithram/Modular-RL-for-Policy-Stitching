# # robot_env = mujoco_env
# # fetch_env = jaco
# # pick_place  = jaco_pick
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import gym
from gym.envs.mujoco.jaco import JacoEnv
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))
import sys

MODEL_XML_PATH = 'jaco_pick.xml'

# for attr in dir(JacoEnv):
#     print("obj.%s = %r" % (attr, getattr(JacoEnv, attr)))

# super(JacoPickEnv,self).__init__(MODEL_XML_PATH, frame_skip=4, target_offset=0.0, obj_range=0.15,
#                 target_range=0.15, distance_threshold=0.05,has_object=True)
# JacoEnv.__init__(
#     self, MODEL_XML_PATH, frame_skip=4, target_offset=0.0, obj_range=0.15,
#     target_range=0.15, distance_threshold=0.05,has_object=True)
# utils.EzPickle.__init__(self)
# super(JacoEnv).__init__(self,MODEL_XML_PATH, frame_skip=4, target_offset=0.0, obj_range=0.15,
#                 target_range=0.15, distance_threshold=0.05,has_object=True)

class JacoPickEnv(JacoEnv, utils.EzPickle):
    def __init__(self, with_rot=1):
        initial_qpos = {
                'jaco_joint_finger_1':0.405,
                'jaco_joint_finger_1':0.41,
                'jaco_joint_finger_1':0.40,
            
        }
        # initial_qpos = {
        #     'robot0:slide0': 0.405,
        #     'robot0:slide1': 0.48,
        #     'robot0:slide2': 0.0,
        #     'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        # }
        super(JacoPickEnv,self).__init__(MODEL_XML_PATH, frame_skip=4, target_offset=0.0, obj_range=0.15,
                        target_range=0.15, distance_threshold=0.05,initial_qpos = initial_qpos,target_in_the_air=False,
                        gripper_extra_height=0.2,block_gripper=False)
        utils.EzPickle.__init__(self)
