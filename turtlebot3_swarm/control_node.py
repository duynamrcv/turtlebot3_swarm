#! /usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np

from mpc_controller import MPC
from utils import *

class ControlState:
    IDLE = 1
    TRACKING = 2
    FINISHING = 3

class MPCNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.rate = 10  # Hz
        self.name = self.declare_parameter('name', 'turtlebot3').get_parameter_value().string_value
        self.max_v = self.declare_parameter('max_v', 0.3).get_parameter_value().double_value
        self.max_w = self.declare_parameter('max_w', np.pi/3).get_parameter_value().double_value
        self.N = self.declare_parameter('horizon', 10).get_parameter_value().integer_value
        self.threshold = self.declare_parameter('goal_threshold', 0.1).get_parameter_value().double_value
        
        self.subscriber_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, self.rate)
        self.subscriber_odometry = self.create_subscription(Odometry, '{}/odom'.format(self.name), self.odom_callback, self.rate)

        self.publisher_vel = self.create_publisher(Twist, '{}/cmd_vel'.format(self.name), self.rate)
        self.publisher_local_path = self.create_publisher(Path, '{}/local_path'.format(self.name), self.rate)

        self.timer = self.create_timer(1.0/self.rate, self.timer_callback)
        self.mpc = MPC(n_states=3, n_action=2, 
                       max_v=self.max_v, max_w=self.max_w, 
                       hoziron_length=self.N, time_step=1.0/self.rate)
        
        self.frame_id = ""
        self.goal = None
        self.pose = None
        self.state = ControlState.IDLE

    def goal_callback(self, msg:PoseStamped):
        self.get_logger().info("Received new assignment")
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        _, _, gtheta = euler_from_quaternion(msg.pose.orientation)
        self.goal = np.array([gx, gy, gtheta])

    def odom_callback(self, msg:Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        _, _, ptheta = euler_from_quaternion(msg.pose.pose.orientation)
        self.pose = np.array([px, py, ptheta])
        self.frame_id = msg.header.frame_id

    def timer_callback(self):
        states = None    
        if self.state == ControlState.IDLE:
            action = self.handle_idle_state()
        elif self.state == ControlState.TRACKING:
            action, states = self.handle_tracking_state()
        else: # ControlState.FINISHING:
            action = self.handle_finish_state()

        self.publish_control_command(action)
        self.publish_local_path(states)

    def publish_control_command(self, action=np.array):
        # Send control signal
        msg_vel = Twist()
        msg_vel.linear.x = action[0]
        msg_vel.angular.z = action[1]
        self.publisher_vel.publish(msg_vel)
    
    def publish_local_path(self, states):
        path = Path()
        path.header.frame_id = self.frame_id
        if states is not None:
            length = states.shape[1]
            for i in range(length):
                state = states[:,i]
                pose = PoseStamped()
                pose.pose.position.x = state[0]
                pose.pose.position.y = state[1]
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)

        self.publisher_local_path.publish(path)

    def handle_idle_state(self):
        self.get_logger().info("Handle IDLE state")
        if self.goal is not None:
            self.get_logger().info("Moving to the target at x={}, y={}, \
                                   theta={}".format(self.goal[0], self.goal[1], self.goal[2]))
            self.state = ControlState.TRACKING
        action = np.array([0., 0.])
        return action
    
    def handle_tracking_state(self):
        self.get_logger().info("Handle TRACKING state")
        if np.linalg.norm(self.goal - self.pose) < self.threshold:
            action = np.array([0., 0.])
            states = None
            self.state = ControlState.FINISHING
        else:
            action, states = self.mpc.compute_action(self.pose, self.goal)
        return action, states
    
    def handle_finish_state(self):
        self.get_logger().info("Handle FINISHING state")
        self.goal = None    # Reset goal to None for new asignment
        self.state = ControlState.IDLE
        action = np.array([0., 0.])
        return action


if __name__ == '__main__':
    rclpy.init(args=None)
    controller = MPCNode()
    rclpy.spin(controller)

    controller.destroy_node()
    rclpy.shutdown()

