cmake_minimum_required(VERSION 3.8)
project(turtlebot3_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Generate the compile_commands.json file
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)

# Install Python modules
ament_python_install_package("src")

# Install Python executables
file(GLOB_RECURSE PYTHON_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.py")
foreach(py_file ${PYTHON_FILES})
  execute_process(COMMAND "chmod" "+x" "${py_file}")
endforeach()

install(PROGRAMS
  ${PYTHON_FILES}
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()
