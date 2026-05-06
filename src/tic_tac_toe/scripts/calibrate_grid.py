#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate_grid.py
棋盘位置标定工具
用于手动标定棋盘格子、取棋区、观察位的位置和关节角度

标定时请确保：
- 每个格子标定表面位（夹爪朝下，末端贴近棋盘表面）
- 取棋区标定表面位（夹爪朝下，末端贴近取棋区表面）
- 观察位标定摄像头能看全棋盘的位置

标定数据说明：
- z坐标表示棋盘表面高度（棋子底部所在平面）
- joints表示表面位的关节角度
- 控制代码通过height_offset从表面位计算实际运动高度
"""

import copy
import rospy
import moveit_commander
from geometry_msgs.msg import PoseStamped
import sys
import os
import yaml


class GridCalibration:
    # 高度参数（与控制代码保持一致）
    HOVER_HEIGHT = 0.06    # 悬停高度
    SAFE_HEIGHT = 0.10     # 安全高度
    PIECE_HEIGHT = 0.024   # 棋子高度（2.4cm）

    @property
    def CALIBRATION_FILE(self):
        import rospkg
        return os.path.join(
            rospkg.RosPack().get_path('tic_tac_toe'),
            'config', 'calibration_data.yaml'
        )

    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('grid_calibration')

        self.arm = moveit_commander.MoveGroupCommander("sagittarius_arm", robot_description="/sgr532/robot_description", ns="/sgr532")
        self.end_effector_link = self.arm.get_end_effector_link()
        self.arm.set_pose_reference_frame('world')

        self.current_grid = {}       # {(row,col): {'x':, 'y':, 'z':, 'joints': []}}
        self.storage_data = None     # {'x':, 'y':, 'z':, 'joints': []}
        self.observation_data = None # {'x':, 'y':, 'z':, 'joints': []}
        self.data_changed = False    # 跟踪数据是否被修改

    def get_current_position(self):
        pose = self.arm.get_current_pose(self.end_effector_link).pose
        return (pose.position.x, pose.position.y, pose.position.z)

    def get_current_joints(self):
        return list(self.arm.get_current_joint_values())

    def move_and_record(self, row, col):
        """
        记录当前机械臂位置到指定格子
        用户应将机械臂拖到棋盘表面位置
        """
        rospy.sleep(0.5)
        x, y, z = self.get_current_position()
        joints = self.get_current_joints()
        self.current_grid[(row, col)] = {
            'x': x, 'y': y, 'z': z,
            'joints': joints
        }
        self.data_changed = True
        rospy.loginfo(f"格子 ({row}, {col}) 标定完成:")
        rospy.loginfo(f"  位置: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        rospy.loginfo(f"  关节: {[f'{j:.4f}' for j in joints]}")

    def calibrate_storage(self):
        """标定取棋区表面位"""
        rospy.loginfo("标定取棋区表面位...")
        rospy.loginfo("请拖动机械臂到取棋区表面（夹爪朝下，末端贴近取棋区表面），输入 'y' 确认")

        try:
            user_input = raw_input("\n取棋区标定> ")
        except NameError:
            user_input = input("\n取棋区标定> ")

        if user_input.lower() == 'y':
            rospy.sleep(0.5)
            x, y, z = self.get_current_position()
            joints = self.get_current_joints()
            self.storage_data = {
                'x': x, 'y': y, 'z': z,
                'joints': joints
            }
            self.data_changed = True
            rospy.loginfo(f"取棋区标定完成:")
            rospy.loginfo(f"  位置: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            rospy.loginfo(f"  关节: {[f'{j:.4f}' for j in joints]}")
            return True
        else:
            rospy.loginfo("取棋区标定取消")
            return False

    def calibrate_observation(self):
        """标定观察位"""
        rospy.loginfo("标定观察位...")
        rospy.loginfo("请拖动机械臂到摄像头能看全棋盘的位置，输入 'y' 确认")

        try:
            user_input = raw_input("\n观察位标定> ")
        except NameError:
            user_input = input("\n观察位标定> ")

        if user_input.lower() == 'y':
            rospy.sleep(0.5)
            joints = self.get_current_joints()
            x, y, z = self.get_current_position()
            self.observation_data = {
                'x': x, 'y': y, 'z': z,
                'joints': joints
            }
            self.data_changed = True
            rospy.loginfo(f"观察位标定完成:")
            rospy.loginfo(f"  位置: x={x:.4f}, y={y:.4f}, z={z:.4f}")
            rospy.loginfo(f"  关节: {[f'{j:.4f}' for j in joints]}")
            return True
        else:
            rospy.loginfo("观察位标定取消")
            return False

    def validate_data(self):
        """验证标定数据的合理性"""
        issues = []

        # 检查格子完整性
        missing = []
        for r in range(3):
            for c in range(3):
                if (r, c) not in self.current_grid:
                    missing.append(f"({r},{c})")
        if missing:
            issues.append(f"未标定格子: {', '.join(missing)}")

        if not self.storage_data:
            issues.append("取棋区未标定")

        if not self.observation_data:
            issues.append("观察位未标定")

        # 检查z值一致性
        z_values = [d['z'] for d in self.current_grid.values()]
        if z_values:
            z_avg = sum(z_values) / len(z_values)
            z_range = max(z_values) - min(z_values)
            if z_range > 0.01:  # 1cm容差
                issues.append(f"格子z值偏差较大: {z_range*1000:.1f}mm (建议<10mm)")
            rospy.loginfo(f"  z值统计: 平均={z_avg*1000:.1f}mm, 范围={z_range*1000:.1f}mm")

        # 检查格子间距（应约5cm）
        for r in range(3):
            for c in range(2):
                if (r, c) in self.current_grid and (r, c+1) in self.current_grid:
                    p1 = self.current_grid[(r, c)]
                    p2 = self.current_grid[(r, c+1)]
                    dx = p2['x'] - p1['x']
                    dy = p2['y'] - p1['y']
                    dist = (dx**2 + dy**2) ** 0.5
                    if abs(dist - 0.05) > 0.02:
                        issues.append(f"格子({r},{c})-({r},{c+1})间距异常: {dist*100:.1f}cm (预期5cm)")

        for c in range(3):
            for r in range(2):
                if (r, c) in self.current_grid and (r+1, c) in self.current_grid:
                    p1 = self.current_grid[(r, c)]
                    p2 = self.current_grid[(r+1, c)]
                    dx = p2['x'] - p1['x']
                    dy = p2['y'] - p1['y']
                    dist = (dx**2 + dy**2) ** 0.5
                    if abs(dist - 0.05) > 0.02:
                        issues.append(f"格子({r},{c})-({r+1},{c})间距异常: {dist*100:.1f}cm (预期5cm)")

        # 检查关节数据
        no_joints = [f"({r},{c})" for (r, c), d in self.current_grid.items() if not d.get('joints')]
        if no_joints:
            issues.append(f"缺少关节数据: {', '.join(no_joints)} (将使用笛卡尔回退)")

        return issues

    def save_to_yaml(self):
        """保存标定结果到YAML文件"""
        # 保存前验证
        issues = self.validate_data()
        if issues:
            rospy.logwarn("标定数据存在问题:")
            for issue in issues:
                rospy.logwarn(f"  - {issue}")
            try:
                user_input = raw_input("\n是否仍要保存? (y/n): ")
            except NameError:
                user_input = input("\n是否仍要保存? (y/n): ")
            if user_input.lower() != 'y':
                rospy.loginfo("保存已取消")
                return

        config_dir = os.path.dirname(self.CALIBRATION_FILE)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        grid_data = {}
        for row in range(3):
            for col in range(3):
                if (row, col) in self.current_grid:
                    d = self.current_grid[(row, col)]
                    grid_data[f"({row}, {col})"] = {
                        "x": round(d['x'], 6),
                        "y": round(d['y'], 6),
                        "z": round(d['z'], 6),
                        "joints": [round(j, 6) for j in d['joints']]
                    }
                else:
                    grid_data[f"({row}, {col})"] = {"x": 0.0, "y": 0.0, "z": 0.0, "joints": []}

        storage = {"x": 0.0, "y": 0.0, "z": 0.0, "joints": []}
        if self.storage_data:
            storage = {
                "x": round(self.storage_data['x'], 6),
                "y": round(self.storage_data['y'], 6),
                "z": round(self.storage_data['z'], 6),
                "joints": [round(j, 6) for j in self.storage_data['joints']]
            }

        observation = {"x": 0.0, "y": 0.0, "z": 0.0, "joints": []}
        if self.observation_data:
            observation = {
                "x": round(self.observation_data['x'], 6),
                "y": round(self.observation_data['y'], 6),
                "z": round(self.observation_data['z'], 6),
                "joints": [round(j, 6) for j in self.observation_data['joints']]
            }

        calibration_data = {
            "grid_positions": grid_data,
            "storage_position": storage,
            "observation_position": observation,
            "metadata": {
                "description": "井字棋游戏标定结果",
                "units": "meters / radians",
                "frame": "world",
                "note": "z坐标为棋盘表面高度，joints为表面位关节角度，控制代码通过height_offset计算实际运动高度",
                "height_config": {
                    "hover_height": self.HOVER_HEIGHT,
                    "safe_height": self.SAFE_HEIGHT,
                    "grasp_height": self.PIECE_HEIGHT / 2,
                    "place_height": self.PIECE_HEIGHT / 2,
                    "piece_height": self.PIECE_HEIGHT
                }
            }
        }

        with open(self.CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(calibration_data, f, allow_unicode=True, default_flow_style=False)

        self.data_changed = False
        rospy.loginfo(f"标定结果已保存到: {self.CALIBRATION_FILE}")

    def output_config(self):
        """输出标定结果到终端"""
        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("标定结果:")
        rospy.loginfo("=" * 60)
        for row in range(3):
            for col in range(3):
                if (row, col) in self.current_grid:
                    d = self.current_grid[(row, col)]
                    has_joints = bool(d.get('joints'))
                    print(f"  ({row},{col}): pos=({d['x']:.4f},{d['y']:.4f},{d['z']:.4f}) joints={'OK' if has_joints else 'MISSING'}")
                else:
                    print(f"  ({row},{col}): 未标定")
        if self.storage_data:
            print(f"  取棋区: pos=({self.storage_data['x']:.4f},{self.storage_data['y']:.4f},{self.storage_data['z']:.4f}) joints={'OK' if self.storage_data.get('joints') else 'MISSING'}")
        else:
            print("  取棋区: 未标定")
        if self.observation_data:
            print(f"  观察位: pos=({self.observation_data['x']:.4f},{self.observation_data['y']:.4f},{self.observation_data['z']:.4f}) joints={'OK' if self.observation_data.get('joints') else 'MISSING'}")
        else:
            print("  观察位: 未标定")

        # 显示验证结果
        issues = self.validate_data()
        if issues:
            rospy.logwarn("数据问题:")
            for issue in issues:
                rospy.logwarn(f"  - {issue}")
        else:
            rospy.loginfo("数据验证通过")
        rospy.loginfo("=" * 60)

    def test_position(self, row, col):
        """测试移动到指定格子"""
        if (row, col) not in self.current_grid:
            rospy.logerr(f"格子 ({row}, {col}) 未标定")
            return

        d = self.current_grid[(row, col)]
        joints = d.get('joints', [])

        rospy.loginfo(f"测试移动到格子 ({row}, {col}) 安全高度...")

        if joints:
            # 先移到表面位
            rospy.loginfo("移动到表面位...")
            self.arm.set_start_state_to_current_state()
            self.arm.set_joint_value_target(joints)
            plan_success, traj, _, _ = self.arm.plan()
            if plan_success:
                self.arm.execute(traj, wait=True)
                rospy.loginfo("已到达表面位")
                # 再上升到安全高度
                rospy.loginfo("上升到安全高度...")
                self.arm.set_start_state_to_current_state()
                current_pose = self.arm.get_current_pose(self.end_effector_link)
                target_pose = copy.deepcopy(current_pose)
                target_pose.pose.position.z += self.SAFE_HEIGHT
                self.arm.set_pose_target(target_pose, self.end_effector_link)
                plan_success, traj, _, _ = self.arm.plan()
                if plan_success:
                    self.arm.execute(traj, wait=True)
                    rospy.loginfo(f"已到达安全高度 ({self.SAFE_HEIGHT*100:.0f}cm)")
                else:
                    rospy.logwarn("上升到安全高度失败")
            else:
                rospy.logerr("关节角度移动失败")
        else:
            rospy.logwarn("无关节角度数据，无法测试")


def main():
    rospy.loginfo("=" * 60)
    rospy.loginfo("棋盘位置标定工具")
    rospy.loginfo("=" * 60)
    rospy.loginfo("标定说明:")
    rospy.loginfo("  每个位置请拖动机械臂到棋盘表面位置")
    rospy.loginfo("  （夹爪朝下，末端贴近棋盘/取棋区表面）")
    rospy.loginfo("  程序同时记录笛卡尔坐标和关节角度")
    rospy.loginfo("  z坐标代表棋盘表面高度，控制代码自动计算运动高度")
    rospy.loginfo("=" * 60)
    rospy.loginfo("操作命令:")
    rospy.loginfo("  行 列  - 标定格子表面位，如: 0 0")
    rospy.loginfo("  t      - 标定取棋区表面位")
    rospy.loginfo("  o      - 标定观察位")
    rospy.loginfo("  v      - 验证标定数据")
    rospy.loginfo("  x 行列 - 测试移动到格子，如: x 0 0")
    rospy.loginfo("  s      - 显示标定结果")
    rospy.loginfo("  w      - 保存标定结果到YAML")
    rospy.loginfo("  q      - 退出")
    rospy.loginfo("=" * 60)

    calib = GridCalibration()

    while not rospy.is_shutdown():
        try:
            user_input = raw_input("\n标定> ")
        except NameError:
            user_input = input("\n标定> ")

        cmd = user_input.strip()

        if cmd.lower() == 'q':
            if calib.data_changed:
                try:
                    save_input = raw_input("数据已修改，是否保存? (y/n): ")
                except NameError:
                    save_input = input("数据已修改，是否保存? (y/n): ")
                if save_input.lower() == 'y':
                    calib.save_to_yaml()
            break
        elif cmd.lower() == 's':
            calib.output_config()
            continue
        elif cmd.lower() == 'w':
            calib.save_to_yaml()
            continue
        elif cmd.lower() == 'v':
            issues = calib.validate_data()
            if issues:
                rospy.logwarn("标定数据问题:")
                for issue in issues:
                    rospy.logwarn(f"  - {issue}")
            else:
                rospy.loginfo("标定数据验证通过")
            continue
        elif cmd.lower() == 't':
            calib.calibrate_storage()
            continue
        elif cmd.lower() == 'o':
            calib.calibrate_observation()
            continue

        # 处理 "x 行 列" 测试命令
        parts = cmd.split()
        if parts[0].lower() == 'x' and len(parts) == 3:
            try:
                row, col = int(parts[1]), int(parts[2])
                if 0 <= row <= 2 and 0 <= col <= 2:
                    calib.test_position(row, col)
                else:
                    rospy.logerr("行列值必须在0-2之间")
            except ValueError:
                rospy.logerr("格式: x 行 列 (如: x 0 1)")
            continue

        # 处理 "行 列" 标定命令
        try:
            parts = cmd.split()
            if len(parts) != 2:
                raise ValueError("需要两个数字")

            row, col = int(parts[0]), int(parts[1])

            if row < 0 or row > 2 or col < 0 or col > 2:
                raise ValueError("行列值必须在0-2之间")

            calib.move_and_record(row, col)

        except ValueError as e:
            rospy.logerr(f"输入无效: {e}")
            rospy.loginfo("格式: 行 列 (如: 0 1), 或 x 行 列 (测试)")


if __name__ == "__main__":
    main()
