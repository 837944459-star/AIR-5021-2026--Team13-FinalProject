cmake_minimum_required(VERSION 3.0.2)
project(tic_tac_toe)

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
)

# 安装Python脚本
catkin_install_python(PROGRAMS
    scripts/tic_tac_toe_vision.py
    scripts/tic_tac_toe_logic.py
    scripts/tic_tac_toe_control.py
    scripts/tic_tac_toe_main.py
    scripts/calibrate_grid.py
    scripts/camera_capture.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch文件
install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装配置文件
install(DIRECTORY config/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
