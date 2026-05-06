#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tic_tac_toe_control.py
机械臂MoveIt!控制模块

优先使用标定的关节角度运动（无需IK，100%可达），
垂直升降使用笛卡尔路径规划。

标定数据说明：
- z坐标为棋盘表面高度（棋子底部所在平面）
- joints为表面位的关节角度
- 所有height_offset从表面位开始计算
"""

import copy
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped, Pose
import sys
import os
import yaml


class TicTacToeControl:
    # 笛卡尔坐标（仅在无关节角度时作为回退）
    GRID_POSITIONS = {
        (0, 0): (-0.05,  0.05, 0.05),
        (0, 1): ( 0.00,  0.05, 0.05),
        (0, 2): ( 0.05,  0.05, 0.05),
        (1, 0): (-0.05,  0.00, 0.05),
        (1, 1): ( 0.00,  0.00, 0.05),
        (1, 2): ( 0.05,  0.00, 0.05),
        (2, 0): (-0.05, -0.05, 0.05),
        (2, 1): ( 0.00, -0.05, 0.05),
        (2, 2): ( 0.05, -0.05, 0.05),
    }

    STORAGE_POSITION = (0.092, 0.05, 0.05)

    # 高度偏移配置（相对于棋盘表面）
    HOVER_HEIGHT = 0.06    # 悬停高度（高于棋子顶部）
    SAFE_HEIGHT = 0.10     # 安全上升高度
    PIECE_HEIGHT = 0.024   # 棋子高度（2.4cm正方体）
    GRASP_HEIGHT = 0.012   # 抓取高度（棋子中心 = 棋子高度/2）
    PLACE_HEIGHT = 0.012   # 放置高度（棋子中心，底部刚好接触棋盘表面）

    def __init__(self, arm_group_name="sagittarius_arm", gripper_group_name="sagittarius_gripper"):
        moveit_commander.roscpp_initialize(sys.argv)

        try:
            rospy.init_node('tic_tac_toe_control')
        except rospy.exceptions.ROSException:
            pass

        self.arm = moveit_commander.MoveGroupCommander(arm_group_name, robot_description="/sgr532/robot_description", ns="/sgr532")
        self.gripper = moveit_commander.MoveGroupCommander(gripper_group_name, robot_description="/sgr532/robot_description", ns="/sgr532")

        self.end_effector_link = self.arm.get_end_effector_link()
        self.arm.set_pose_reference_frame('world')

        self.arm.set_goal_position_tolerance(0.005)
        self.arm.set_goal_orientation_tolerance(0.1)
        self.gripper.set_goal_joint_tolerance(0.001)

        self.arm.set_max_velocity_scaling_factor(0.3)
        self.arm.set_max_acceleration_scaling_factor(0.3)
        self.gripper.set_max_velocity_scaling_factor(0.5)
        self.gripper.set_max_acceleration_scaling_factor(0.5)

        self.arm.allow_replanning(True)

        # 关节角度数据（从标定文件加载，代表表面位）
        self.GRID_JOINTS = {}
        self.STORAGE_JOINTS = None
        self.OBSERVATION_JOINTS = None
        self.OBSERVATION_POSITION = None  # 从标定数据加载

        self._load_calibration()

        rospy.loginfo("MoveIt! 控制初始化完成")
        rospy.loginfo(f"末端link: {self.end_effector_link}")
        rospy.loginfo(f"关节角度数据: 格子{len(self.GRID_JOINTS)}个, "
                      f"取棋区{'有' if self.STORAGE_JOINTS else '无'}, "
                      f"观察位{'有' if self.OBSERVATION_JOINTS else '无'}")

    # ==================== 基础运动方法 ====================

    def go_to_home(self):
        rospy.loginfo("机械臂返回Home位置...")
        self.arm.set_named_target('home')
        self.arm.go()
        rospy.sleep(1)

    def go_to_sleep(self):
        rospy.loginfo("机械臂返回Sleep位置...")
        self.arm.set_named_target('sleep')
        self.arm.go()
        rospy.sleep(1)

    def open_gripper(self):
        rospy.loginfo("张开夹爪...")
        self.gripper.set_named_target('open')
        self.gripper.go()
        rospy.sleep(1)

    def close_gripper(self):
        rospy.loginfo("闭合夹爪...")
        self.gripper.set_named_target('close')
        self.gripper.go()
        rospy.sleep(1)

    # ==================== 关节角度运动（核心方法） ====================

    def _move_to_joint_target(self, joint_values):
        """用关节角度规划运动，无需IK，100%可达"""
        self.arm.set_start_state_to_current_state()
        try:
            self.arm.set_joint_value_target(joint_values)
        except Exception as e:
            rospy.logerr(f"关节目标超出范围: {e}")
            return False

        plan_success, traj, planning_time, error_code = self.arm.plan()

        if plan_success:
            rospy.loginfo(f"关节规划成功，轨迹点数: {len(traj.joint_trajectory.points)}")
            self.arm.execute(traj, wait=True)
            return True
        else:
            rospy.logerr(f"关节规划失败, error_code: {error_code}")
            return False

    def _cartesian_move(self, dz):
        """沿Z轴直线移动dz米（正=上，负=下），保持当前朝向"""
        waypoints = []
        current_pose = self.arm.get_current_pose(self.end_effector_link)
        target_pose = copy.deepcopy(current_pose.pose)  # 提取Pose，compute_cartesian_path需要Pose而非PoseStamped
        target_pose.position.z += dz
        waypoints.append(target_pose)

        (plan, fraction) = self.arm.compute_cartesian_path(waypoints, 0.005, 0.0)

        if fraction < 0.9:
            rospy.logerr(f"笛卡尔路径规划失败 (完成率: {fraction:.0%})")
            return False

        rospy.loginfo(f"笛卡尔移动 {dz*100:.1f}cm (完成率: {fraction:.0%})")
        self.arm.execute(plan, wait=True)
        return True

    # ==================== 位置移动方法 ====================

    def go_to_observation(self):
        """移动到观察位"""
        rospy.loginfo("移动到观察位...")
        if self.OBSERVATION_JOINTS:
            if self._move_to_joint_target(self.OBSERVATION_JOINTS):
                return True
            rospy.logwarn("观察位关节角度超限，尝试笛卡尔规划回退...")
        if self.OBSERVATION_POSITION:
            x, y, z = self.OBSERVATION_POSITION
            self.arm.set_start_state_to_current_state()
            self.arm.set_position_target([x, y, z], self.end_effector_link)
            plan_success, traj, _, error_code = self.arm.plan()
            if plan_success:
                self.arm.execute(traj, wait=True)
                return True
            rospy.logerr(f"观察位移动失败: {error_code}")
            return False
        else:
            rospy.logerr("无观察位数据，请先标定")
            return False

    def move_to_grid(self, row, col, height_offset=0):
        """
        移动到指定格子，优先用关节角度到表面位，再经悬停位安全调整

        安全策略：始终先上升到悬停位（高于棋子），再调整到目标高度，
        避免关节空间运动时夹爪经过已有棋子的高度。

        Args:
            row: 行号 (0-2)
            col: 列号 (0-2)
            height_offset: 目标高度相对棋盘表面的偏移（米）
                           GRASP_HEIGHT=0.012, HOVER_HEIGHT=0.06, SAFE_HEIGHT=0.10
        """
        if (row, col) in self.GRID_JOINTS:
            # Step 1: 关节角度到表面位（精确定位xy）
            if not self._move_to_joint_target(self.GRID_JOINTS[(row, col)]):
                return False
            # Step 2: 上升到悬停位（安全高度，高于所有棋子）
            if not self._cartesian_move(self.HOVER_HEIGHT):
                rospy.logwarn("上升至悬停位失败，停留在表面位")
                return False
            # Step 3: 从悬停位调整到目标高度
            dz = height_offset - self.HOVER_HEIGHT
            if abs(dz) > 0.002:
                if not self._cartesian_move(dz):
                    rospy.logwarn(f"高度调整失败(dz={dz:.3f})，停留在悬停位")
            return True
        else:
            # 回退：笛卡尔规划
            if (row, col) not in self.GRID_POSITIONS:
                rospy.logerr(f"无效的格子位置: ({row}, {col})")
                return False
            x, y, z = self.GRID_POSITIONS[(row, col)]
            return self._move_to_position(x, y, z, height_offset)

    def move_to_storage(self, height_offset=0):
        """
        移动到取棋区，优先用关节角度，经悬停位安全调整

        Args:
            height_offset: 目标高度相对取棋区表面的偏移（米）
        """
        if self.STORAGE_JOINTS:
            if not self._move_to_joint_target(self.STORAGE_JOINTS):
                return False
            # 先上升到悬停位
            if not self._cartesian_move(self.HOVER_HEIGHT):
                rospy.logwarn("上升至悬停位失败，停留在表面位")
                return False
            # 从悬停位调整到目标高度
            dz = height_offset - self.HOVER_HEIGHT
            if abs(dz) > 0.002:
                if not self._cartesian_move(dz):
                    rospy.logwarn(f"高度调整失败(dz={dz:.3f})，停留在悬停位")
            return True
        else:
            x, y, z = self.STORAGE_POSITION
            rospy.loginfo(f"移动到取棋区: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            return self._move_to_position(x, y, z, height_offset)

    def _move_to_position(self, x, y, z, height_offset=0):
        """笛卡尔位姿规划（回退方法）"""
        target_z = z + height_offset
        rospy.loginfo(f"规划移动到位置: x={x:.4f}, y={y:.4f}, z={target_z:.4f}")

        target_pose = PoseStamped()
        target_pose.header.frame_id = 'world'
        target_pose.header.stamp = rospy.Time.now()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = target_z
        target_pose.pose.orientation.w = 0.682
        target_pose.pose.orientation.x = 0.001
        target_pose.pose.orientation.y = 0.731
        target_pose.pose.orientation.z = 0.001

        self.arm.set_start_state_to_current_state()
        self.arm.set_pose_target(target_pose, self.end_effector_link)

        plan_success, traj, _, error_code = self.arm.plan()
        if plan_success:
            self.arm.execute(traj, wait=True)
            return True

        rospy.logwarn("位姿规划失败，尝试只约束位置...")
        self.arm.set_start_state_to_current_state()
        self.arm.set_position_target([x, y, target_z], self.end_effector_link)

        plan_success, traj, _, error_code = self.arm.plan()
        if plan_success:
            self.arm.execute(traj, wait=True)
            return True

        rospy.logerr(f"移动失败, error_code: {error_code}")
        return False

    # ==================== 抓取/放置动作序列 ====================

    def place_piece(self, row, col):
        """
        放置棋子动作序列（夹爪已夹着棋子）：
        0. 移到目标格子上方安全高度（避免水平移动时撞到已有棋子）
        1. 下降到悬停高度
        2. 下降到放置高度（棋子底部接触棋盘表面）
        3. 张开夹爪释放棋子
        4. 上升到安全高度
        """
        rospy.loginfo(f"=== 开始放置棋子到 ({row}, {col}) ===")

        try:
            # 0. 先移到安全高度（保证水平移动时有足够高度避免碰撞）
            rospy.loginfo("步骤0: 移动到安全高度...")
            if not self.move_to_grid(row, col, height_offset=self.SAFE_HEIGHT):
                return False

            # 1. 下降到悬停高度
            rospy.loginfo("步骤1: 下降到悬停高度...")
            descent = self.SAFE_HEIGHT - self.HOVER_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.GRID_POSITIONS[(row, col)]
                if not self._move_to_position(x, y, z, height_offset=self.HOVER_HEIGHT):
                    return False
            rospy.sleep(0.3)

            # 2. 下降到放置高度
            rospy.loginfo("步骤2: 下降到放置高度...")
            descent = self.HOVER_HEIGHT - self.PLACE_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.GRID_POSITIONS[(row, col)]
                if not self._move_to_position(x, y, z, height_offset=self.PLACE_HEIGHT):
                    return False
            rospy.sleep(0.5)

            # 3. 松开棋子
            rospy.loginfo("步骤3: 张开夹爪释放棋子...")
            self.open_gripper()
            rospy.sleep(0.5)

            # 4. 上升到安全高度
            rospy.loginfo("步骤4: 上升到安全高度...")
            ascent = self.SAFE_HEIGHT - self.PLACE_HEIGHT
            if not self._cartesian_move(ascent):
                self.go_to_home()

            rospy.loginfo(f"=== 棋子放置到 ({row}, {col}) 完成 ===")
            return True

        except Exception as e:
            rospy.logerr(f"放置棋子时发生异常: {e}")
            return False

    def pick_piece(self, row, col):
        """
        抓取棋子动作序列：
        0. 张开夹爪
        1. 移到目标格子上方安全高度
        2. 下降到悬停高度
        3. 下降到抓取高度
        4. 闭合夹爪抓取
        5. 上升到安全高度
        """
        rospy.loginfo(f"=== 开始抓取棋子从 ({row}, {col}) ===")

        try:
            # 0. 张开夹爪准备抓取
            self.open_gripper()
            rospy.sleep(0.5)

            # 1. 移到安全高度
            rospy.loginfo("步骤1: 移动到安全高度...")
            if not self.move_to_grid(row, col, height_offset=self.SAFE_HEIGHT):
                return False

            # 2. 下降到悬停高度
            rospy.loginfo("步骤2: 下降到悬停高度...")
            descent = self.SAFE_HEIGHT - self.HOVER_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.GRID_POSITIONS[(row, col)]
                if not self._move_to_position(x, y, z, height_offset=self.HOVER_HEIGHT):
                    return False
            rospy.sleep(0.3)

            # 3. 下降到抓取高度
            rospy.loginfo("步骤3: 下降到抓取高度...")
            descent = self.HOVER_HEIGHT - self.GRASP_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.GRID_POSITIONS[(row, col)]
                if not self._move_to_position(x, y, z, height_offset=self.GRASP_HEIGHT):
                    return False
            rospy.sleep(0.5)

            # 4. 闭合夹爪抓取
            rospy.loginfo("步骤4: 闭合夹爪抓取...")
            self.close_gripper()
            rospy.sleep(0.5)

            # 5. 上升到安全高度
            rospy.loginfo("步骤5: 上升到安全高度...")
            ascent = self.SAFE_HEIGHT - self.GRASP_HEIGHT
            if not self._cartesian_move(ascent):
                self.go_to_home()

            rospy.loginfo(f"=== 从 ({row}, {col}) 抓取棋子完成 ===")
            return True

        except Exception as e:
            rospy.logerr(f"抓取棋子时发生异常: {e}")
            return False

    def pick_piece_from_storage(self):
        """
        从取棋区抓取棋子（深蓝色方块）：
        0. 张开夹爪
        1. 移到取棋区上方安全高度
        2. 下降到悬停高度
        3. 下降到抓取高度
        4. 闭合夹爪抓取
        5. 上升到安全高度
        """
        rospy.loginfo("=== 从取棋区抓取棋子 ===")

        try:
            # 0. 张开夹爪准备抓取
            self.open_gripper()
            rospy.sleep(0.5)

            # 1. 移到安全高度
            rospy.loginfo("步骤1: 移动到取棋区上方安全高度...")
            if not self.move_to_storage(height_offset=self.SAFE_HEIGHT):
                return False

            # 2. 下降到悬停高度
            rospy.loginfo("步骤2: 下降到悬停高度...")
            descent = self.SAFE_HEIGHT - self.HOVER_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.STORAGE_POSITION
                if not self._move_to_position(x, y, z, height_offset=self.HOVER_HEIGHT):
                    return False
            rospy.sleep(0.3)

            # 3. 下降到抓取高度
            rospy.loginfo("步骤3: 下降到抓取高度...")
            descent = self.HOVER_HEIGHT - self.GRASP_HEIGHT
            if not self._cartesian_move(-descent):
                x, y, z = self.STORAGE_POSITION
                if not self._move_to_position(x, y, z, height_offset=self.GRASP_HEIGHT):
                    return False
            rospy.sleep(0.5)

            # 4. 闭合夹爪抓取
            rospy.loginfo("步骤4: 闭合夹爪抓取...")
            self.close_gripper()
            rospy.sleep(0.5)

            # 5. 上升到安全高度
            rospy.loginfo("步骤5: 上升到安全高度...")
            ascent = self.SAFE_HEIGHT - self.GRASP_HEIGHT
            if not self._cartesian_move(ascent):
                self.go_to_home()

            rospy.loginfo("=== 取棋区抓取完成 ===")
            return True

        except Exception as e:
            rospy.logerr(f"从取棋区抓取时发生异常: {e}")
            return False

    # ==================== 标定数据加载 ====================

    def update_grid_positions(self, grid_positions):
        self.GRID_POSITIONS.update(grid_positions)
        rospy.loginfo("更新了棋盘格子位置配置")

    def _load_calibration(self):
        import rospkg
        pkg_path = rospkg.RosPack().get_path('tic_tac_toe')
        calib_file = os.path.join(pkg_path, 'config', 'calibration_data.yaml')

        if not os.path.exists(calib_file):
            rospy.logwarn(f"标定文件不存在: {calib_file}")
            rospy.logwarn("请运行 calibrate_grid.py 进行标定")
            return

        try:
            with open(calib_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 加载格子位置和关节角度
            grid_positions = data.get('grid_positions', {})
            for key, pos in grid_positions.items():
                key = key.strip('()')
                row, col = map(int, key.split(','))
                self.GRID_POSITIONS[(row, col)] = (pos['x'], pos['y'], pos['z'])
                joints = pos.get('joints', [])
                if joints:
                    self.GRID_JOINTS[(row, col)] = joints

            # 加载取棋区
            storage = data.get('storage_position')
            if storage:
                self.STORAGE_POSITION = (storage['x'], storage['y'], storage['z'])
                joints = storage.get('joints', [])
                if joints:
                    self.STORAGE_JOINTS = joints

            # 加载观察位
            observation = data.get('observation_position')
            if observation:
                joints = observation.get('joints', [])
                if joints:
                    self.OBSERVATION_JOINTS = joints
                # 也加载笛卡尔坐标作为回退
                if 'x' in observation and 'y' in observation and 'z' in observation:
                    self.OBSERVATION_POSITION = (observation['x'], observation['y'], observation['z'])

            rospy.loginfo(f"已加载标定数据: {calib_file}")
            for (r, c), (x, y, z) in sorted(self.GRID_POSITIONS.items()):
                has_joints = (r, c) in self.GRID_JOINTS
                rospy.loginfo(f"  ({r},{c}): pos=({x:.4f},{y:.4f},{z:.4f}) joints={'OK' if has_joints else 'NO'}")
            rospy.loginfo(f"  取棋区: pos=({self.STORAGE_POSITION[0]:.4f},{self.STORAGE_POSITION[1]:.4f},{self.STORAGE_POSITION[2]:.4f}) joints={'OK' if self.STORAGE_JOINTS else 'NO'}")
            rospy.loginfo(f"  观察位: joints={'OK' if self.OBSERVATION_JOINTS else 'NO'} pos={'OK' if self.OBSERVATION_POSITION else 'NO'}")

        except Exception as e:
            rospy.logwarn(f"加载标定数据失败，使用默认位置: {e}")

    def get_current_pose(self):
        return self.arm.get_current_pose(self.end_effector_link).pose

    def shutdown(self):
        rospy.loginfo("关闭机械臂控制...")
        self.open_gripper()
        rospy.sleep(0.5)
        self.go_to_home()
        rospy.sleep(1)
        moveit_commander.roscpp_shutdown()


def main():
    rospy.loginfo("初始化机械臂控制模块...")
    control = TicTacToeControl()

    rospy.loginfo("测试: 移动到Home位置...")
    control.go_to_home()

    rospy.loginfo("测试: 张开夹爪...")
    control.open_gripper()

    rospy.loginfo("控制测试完成")


if __name__ == "__main__":
    main()
