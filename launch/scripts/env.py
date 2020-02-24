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

import config

class GazeboEnv(Environment):
    def __init__(self,maze_id=0,continuous=False):
        self.maze_id = maze_id
        self.continuous = continuous
        self.goal_space = config.goal_space[maze_id]
        self.start_space = config.start_space[maze_id]

        # 启动launch文件
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        path = rospkg.RosPack().get_path('nav_rl')  # 'rospack find' python api, find the path of rl_nav package
        self.launch = roslaunch.parent.ROSLaunchParent(uuid, [path + '/launch/nav_gazebo.launch'])
        self.launch.start()

        rospy.init_node('env_node')
        time.sleep(5)  # Wait for gzserver to launch
        # rospy.spin()

        # 输入图片设置
        self.img_height = config.input_dim[0]
        self.img_width =  config.input_dim[1]
        self.img_channels = config.input_dim[2]

        # 重新仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.vel_pub = rospy.Publisher(' /cmd_vel', Twist, queue_size=1)
        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        # Resets the state of the environment and returns an initial observation.
        start_index = np.random.choice(len(self.start_space))
        goal_index = np.random.choice(len(self.goal_space))
        # start_index, goal_index = np.random.choice(len(self.start_space), 2, replace=False)
        start = self.start_space[start_index]
        theta =  np.random.uniform(0, 2.0*math.pi)  # 1.0/2*math.pi  # 4.0/3*math.pi  #
        self.set_start(start[0], start[1], theta)
        self.goal = self.goal_space[goal_index]
        self.set_goal(self.goal[0], self.goal[1])
        d0, alpha0 = self.goal2robot(self.goal[0] - start[0], self.goal[1] - start[1], theta)
        print(d0, alpha0)
        self.p = [d0, alpha0]  # relative target position
        self.reward = 0

        self.success = False
        self.vel_cmd = [0., 0.]

        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # self.unpause
            rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        except rospy.ServiceException:
            print("/gazebo/unpause_physics service call failed")

        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
            # h = image_data.height
            # w = image_data.width
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause
        except rospy.ServiceException:
            print("/gazebo/pause_physics service call failed")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))  # width, height

        observation = cv_image  # .reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return observation

    def execute(self):
        while (True):
            action = np.random.choice(5)
            robot_state = self.get_state("robot", "world")
            pose = robot_state.pose
            vel_cmd = robot_state.twist

            if action == 0:  # Left
                vel_cmd.linear.x = 0.1
                vel_cmd.angular.z = 1.0
            elif action == 1:  # H-LEFT
                vel_cmd.linear.x = 0.1
                vel_cmd.angular.z = 0.4
            elif action == 2:  # Straight
                vel_cmd.linear.x = 0.1
                vel_cmd.angular.z = 0
            elif action == 3:  # H-Right
                vel_cmd.linear.x = 0.1
                vel_cmd.angular.z = -0.4
            elif action == 4:  # Right
                vel_cmd.linear.x = 0.1
                vel_cmd.angular.z = -1.0
            else:
                raise Exception('Error discrete action: {}'.format(action))

            self.vel_pub.publish(vel_cmd)
            state = ModelState()
            state.model_name = 'robot'
            state.reference_frame = 'world'
            state.pose = pose
            state.twist = vel_cmd
            rospy.wait_for_service('/gazebo/set_model_state')
            try:
                result = self.set_state(state)
                print(result)
            except rospy.ServiceException:
                print("/gazebo/get_model_state service call failed")


    def seed(self, seed):
        pass

    def states(self):
        pass

    def actions(self):
        pass

    def close(self):
        pass

    def goal2robot(self, d_x, d_y, theta):
        d = math.sqrt(d_x * d_x + d_y * d_y)
        alpha = math.atan2(d_y, d_x) - theta
        return d, alpha

    def set_start(self,x,y,theta):
        state = ModelState()
        state.model_name = 'robot'
        state.reference_frame = 'world'  # ''ground_plane'

        # 位置
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0

        #姿态
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
        state.pose.orientation.x = quaternion[0]
        state.pose.orientation.y = quaternion[1]
        state.pose.orientation.z = quaternion[2]
        state.pose.orientation.w = quaternion[3]
        # 速度
        state.twist.linear.x = 0
        state.twist.linear.y = 0
        state.twist.linear.z = 0
        state.twist.angular.x = 0
        state.twist.angular.y = 0
        state.twist.angular.z = 0


    def set_goal(self, x, y):
        state = ModelState()
        state.model_name = 'goal'
        state.reference_frame = 'world'  # ''ground_plane'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")