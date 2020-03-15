#coding: utf8
from __future__ import absolute_import
from __future__ import print_function

import rospy
import roslaunch
import rospkg
import tf

from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from sensor_msgs.msg import Image,LaserScan

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

r_collision = config.r_collision


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
        self._states = dict(shape=(self.img_height, self.img_width, self.img_channels), type='float')
        self._actions = dict(num_actions=5, type='int')
        if self.continuous:
            self._actions = dict(linear_vel=dict(shape=(), type='float', min_value=0.0, max_value=1.0),
                                 angular_vel=dict(shape=(), type='float', min_value=-1.0, max_value=1.0))
        self.vel_cmd = [0., 0.]
        # 重新仿真
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

        self.get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # self.pub_cmd_vel = rospy.Publisher("/cmd_vel",Twist,queue_size=5)

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except rospy.ServiceException:
            print("/gazebo/reset_simulation service call failed")

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

        #get image_data
        state = dict()
        image_data = None
        cv_image = None
        while image_data is None:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image)
            # h = image_data.height
            # w = image_data.width
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))  # width, height

        state['img']= cv_image  # .reshape(1, 1, cv_image.shape[0], cv_image.shape[1])

        # get scan_data
        scan_data = None
        while scan_data is None:
            try:
                scan_data = rospy.wait_for_message('/turtlebot/laser/scan', LaserScan, timeout=5)
            except:
                pass

        state['scan'] = self.get_scan_data(scan_data)

        return state

    def execute(self,action):

        robot_state = self.get_state("robot", "world")
        pose = robot_state.pose
        vel_cmd = robot_state.twist
        if self.continuous:
            # print("continuos velocity")
            vel_cmd.linear.x = -config.v_max*action['linear_vel']
            vel_cmd.angular.z = config.w_max*action['angular_vel']
        else:
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

        state = ModelState()
        state.model_name = 'robot'
        state.reference_frame = 'world'
        state.pose = pose
        state.twist = vel_cmd
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            result = self.set_state(state)
            # print(result)
            # time.sleep(0.05)
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

        done = False
        self.reward = 0
        state = dict()
        image_data = None
        scan_data = None
        cv_image = None

        while image_data is None:
            try:
                for i in range(10):
                    image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                pass
        if self.img_channels == 1:
            cv_image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_width, self.img_height))
        state['img'] = cv_image

        while scan_data is None:
            try:
                for i in range(10):
                    scan_data = rospy.wait_for_message('/turtlebot/laser/scan', LaserScan, timeout=5)
            except:
                pass

        state['scan'] = self.get_scan_data(scan_data)


        # contact_data = None
        # while contact_data is None:
        #     contact_data = rospy.wait_for_message("/contact_state",ContactsState,timeout=5)
        # collision = contact_data.states != []
        # if collision:
        #     done = True
        #     self.reward = r_collision
        #     print("collision!")
        # print("collision: ",collision,"contact_data_states: ", contact_data.states)
        robot_state = None
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            get_state = self.get_state  # create a handle for calling the service
            # use the handle just like a normal function, "robot" relative to "world"
            robot_state = get_state("robot", "world")
            assert robot_state.success is True
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

        position = robot_state.pose.position
        orientation = robot_state.pose.orientation
        d_x = self.goal[0] - position.x
        d_y = self.goal[1] - position.y

        _, _, theta = tf.transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        d, alpha = self.goal2robot(d_x, d_y, theta)
        # print(d, alpha)
        min_range = 0.15
        collision = min_range > min(scan_data.ranges) > 0
        if collision:
            done = True
            if d < config.Cd:
                done = True
                self.reward = config.r_arrive
                self.success = True
                print("arrival!")
            else:
                self.reward = r_collision
                print("collision!")

        if not done:
            delta_d = self.p[0] - d
            self.reward = config.Cr * delta_d + config.Cp

        self.p = [d, alpha]

        return state, self.reward, done

    def seed(self, seed):
        return None

    def states(self):
        return self._states

    def actions(self):
        return self._actions

    def close(self):
        self.launch.shutdown()

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

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = self.set_state
            result = set_state(state)
            assert result.success is True
            print("reset robot success")
        except rospy.ServiceException:
            print("/gazebo/get_model_state service call failed")

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

    def get_scan_data(self,scan):
        scan_range = []
        min_range = 0.13
        for i in range(0,len(scan.ranges),25):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        return scan_range