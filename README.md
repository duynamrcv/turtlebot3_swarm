# ROS2 Turtlebot 3 Waffle on Gazebo

### ROS2 Topic
- `<robot_prefix>/robot_description`
- `<robot_prefix>/joint_states`
- `<robot_prefix>/cmd_vel`
- `<robot_prefix>/odom`
- `<robot_prefix>/scan`

## Prerequisites
```
sudo apt install ros-$ROS_DISTRO-gazebo-ros ros-$ROS_DISTRO-gazebo-ros-pkgs
sudo apt install ros-$ROS_DISTRO-xacro
sudo apt install ros-$ROS_DISTRO-turtlebot3-msgs
```

## Installation
```
git clone git@github.com:duynamrcv/turtlebot3_swarm.git
cd turtlebot3_swarm/
colcon build --symlink-install
```

## Demo
### Launch the empty environment with single Waffle robots controlled by MPC
```
ros2 launch turtlebot3_gazebo mpc_test.launch.py 
```
Use rviz to select goal
### Spawn a robot using a robot_prefix in a particular x_pose and y_pose.
```
ros2 launch turtlebot3_gazebo spawn_turtlebot3.launch.py robot_prefix:=robot1 x_pose:=0.5 y_pose:=0.5
```