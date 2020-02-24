#coding: utf8
from __future__ import absolute_import
from __future__ import print_function

import rospy
import roslaunch
import rospkg
import tf

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image

from gazebo_msgs.msg import ContactsState
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import SetModelState

from tensorforce.environments import Environment
import numpy as np
import math
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError

class GazeboEnv(Environment):
    def __init__(self):
        # 启动launch文件
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        path = rospkg.RosPack().get_path('nav_rl')  # 'rospack find' python api, find the path of rl_nav package
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [path + '/launch/nav_gazebo.launch'])
        self.launch.start()

        rospy.init_node('env_node')
        time.sleep(5)  # Wait for gzserver to launch
        rospy.spin()

    def reset(self):
        pass

    def execute(self, actions):
        pass

    def seed(self, seed):
        pass

    def states(self):
        pass

    def actions(self):
        pass

    def close(self):
        pass