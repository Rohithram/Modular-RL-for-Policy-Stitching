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

class JacoPickEnv(JacoEnv, utils.EzPickle):
    def __init__(self, with_rot=1):
        JacoEnv.__init__(
            self, MODEL_XML_PATH, target_offset=0.0, obj_range=0.15, 
            target_range=0.15, distance_threshold=0.05)
        utils.EzPickle.__init__(self)