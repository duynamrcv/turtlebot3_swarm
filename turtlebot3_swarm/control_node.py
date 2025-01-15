#! /usr/bin/env python3

import rclpy
import rclpy.logging
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np

from turtlebot3_swarm.mpc_controller import MPC
from turtlebot3_swarm.utils import *

class ControlState:
    IDLE = 1
    TRACKING = 2
    FINISHING = 3

class MPCNode(Node):
    def __init__(self):
        super().__init__('control_node')
        self.rate = 10  # Hz
        self.name = self.declare_parameter('name', 'turtlebot3').get_parameter_value().string_value
        
        self.subscriber_goal = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, self.rate)
        self.subscriber_odometry = self.create_subscription(Odometry, '{}/odom'.format(self.name), self.odom_callback, self.rate)

        self.publisher_vel = self.create_publisher(Twist, '{}/cmd_vel'.format(self.name), self.rate)
        self.timer = self.create_timer(1.0/self.rate, self.timer_callback)
        self.mpc = MPC(n_states=3, n_action=2, 
                       max_v=0.3, max_w=np.pi/6, 
                       hoziron_length=10, time_step=1.0/self.rate)
        
        self.goal_queue = []
        self.goal = None
        self.pose = None
        self.state = ControlState.IDLE

    def goal_callback(self, msg:PoseStamped):
        self.get_logger().info("Received new assignment")
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        _, _, gtheta = euler_from_quaternion(msg.pose.orientation)
        self.goal_queue.append(np.array([gx, gy, gtheta]))

    def odom_callback(self, msg:Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        _, _, ptheta = euler_from_quaternion(msg.pose.pose.orientation)
        self.pose = np.array([px, py, ptheta])

    def timer_callback(self):        
        if self.state == ControlState.IDLE:
            action = self.handle_idle_state()
        elif self.state == ControlState.TRACKING:
            action = self.handle_tracking_state()
        else: # ControlState.FINISHING:
            action = self.handle_finish_state()

        # Send control signal
        msg_vel = Twist()
        msg_vel.linear.x = action[0]
        msg_vel.angular.z = action[1]

        self.publisher_vel.publish(msg_vel)

    def handle_idle_state(self):
        self.get_logger().info("Handle IDLE state")
        if self.goal is None and self.goal_queue != []:
            self.goal = self.goal_queue.pop(0)
            self.get_logger().info("Moving to the target at x={}, y={}, \
                                   theta={}".format(self.goal[0], self.goal[1], self.goal[2]))
            self.state = ControlState.TRACKING
        action = np.array([0., 0.])
        return action
    
    def handle_tracking_state(self):
        self.get_logger().info("Handle TRACKING state")
        if np.linalg.norm(self.goal - self.pose) < 0.05:
            action = np.array([0., 0.])
            self.state = ControlState.FINISHING
        else:
            action = self.mpc.compute_action(self.pose, self.goal)
        return action
    
    def handle_finish_state(self):
        self.get_logger().info("Handle FINISHING state")
        self.goal = None    # Reset goal to None for new asignment
        self.state = ControlState.IDLE
        action = np.array([0., 0.])
        return action


if __name__ == '__main__':
    try:
        rclpy.init(args=None)
        controller = MPCNode()
        rclpy.spin(controller)

    except:
        controller.destroy_node()
        rclpy.shutdown()

