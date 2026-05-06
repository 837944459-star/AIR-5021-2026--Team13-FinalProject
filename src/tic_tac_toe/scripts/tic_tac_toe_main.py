#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tic_tac_toe_main.py
井字棋游戏主控程序
整合视觉、逻辑和执行模块
集成摄像头获取图像
"""

import rospy
import os
import sys

# 将源码目录加入sys.path，使import直接找到源文件而非relay脚本
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tic_tac_toe_vision import TicTacToeVision
from tic_tac_toe_logic import TicTacToeLogic
from tic_tac_toe_control import TicTacToeControl
from camera_capture import CameraCapture


class TicTacToeGame:
    def __init__(self):
        rospy.init_node('tic_tac_toe_game')

        # 从参数服务器获取配置
        self.api_key = rospy.get_param('~api_key', None)
        self.image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        self.manual_human_move = rospy.get_param('~manual_human_move', True)
        self.auto_capture = rospy.get_param('~auto_capture', True)

        # 初始化各模块
        rospy.loginfo("=" * 50)
        rospy.loginfo("初始化井字棋游戏系统...")
        rospy.loginfo("=" * 50)

        rospy.loginfo("[1/4] 初始化摄像头模块...")
        try:
            self.camera = CameraCapture(image_topic=self.image_topic)
            if self.camera.is_available():
                rospy.loginfo("摄像头初始化成功")
            else:
                rospy.logwarn("摄像头初始化失败，将使用手动输入模式")
                self.manual_human_move = True
        except Exception as e:
            rospy.logerr(f"摄像头初始化异常: {e}")
            rospy.logwarn("将使用手动输入模式")
            self.manual_human_move = True
            self.camera = None

        rospy.loginfo("[2/4] 初始化视觉模块...")
        try:
            self.vision = TicTacToeVision(api_key=self.api_key)
            rospy.loginfo("视觉模块初始化成功")
        except ValueError as e:
            rospy.logerr(f"视觉模块初始化失败: {e}")
            sys.exit(1)

        rospy.loginfo("[3/4] 初始化游戏逻辑模块...")
        try:
            self.logic = TicTacToeLogic(api_key=self.api_key)
            rospy.loginfo("游戏逻辑模块初始化成功")
        except ValueError as e:
            rospy.logerr(f"游戏逻辑模块初始化失败: {e}")
            sys.exit(1)

        rospy.loginfo("[4/4] 初始化机械臂控制模块...")
        self.control = TicTacToeControl()
        rospy.loginfo("机械臂控制模块初始化成功")

        # 初始化棋盘状态
        self.board = [[0] * 3 for _ in range(3)]
        self.logic.update_board(self.board)

        rospy.loginfo("=" * 50)
        rospy.loginfo("井字棋游戏系统初始化完成！")
        rospy.loginfo(f"人类: X (2), 机器人: O (1)")
        rospy.loginfo(f"手动输入模式: {self.manual_human_move}")
        rospy.loginfo(f"自动捕获模式: {self.auto_capture}")
        rospy.loginfo("=" * 50)

    def capture_board_image(self):
        """
        从摄像头捕获棋盘图像

        Returns:
            str: 保存的图像路径，或None如果捕获失败
        """
        if self.camera is None or not self.camera.is_available():
            rospy.logerr("摄像头不可用，无法捕获图像")
            return None

        rospy.loginfo("从摄像头捕获图像...")
        save_path = rospy.get_param('~capture_save_dir', '/tmp') + '/board_capture.jpg'
        image_path = self.camera.capture_image(save_path)

        if image_path:
            rospy.loginfo(f"图像已保存: {image_path}")
        else:
            rospy.logerr("图像捕获失败")

        return image_path

    def capture_and_detect_board(self):
        """
        捕获图像并检测棋盘状态（使用OpenCV数组直接传递）

        Returns:
            list: 3x3棋盘状态，或None如果检测失败
        """
        if self.camera is None or not self.camera.is_available():
            rospy.logerr("摄像头不可用")
            return None
        
        # 先移到观察位再拍照
        rospy.loginfo("移动到观察位拍照...")
        if not self.control.go_to_observation():
            rospy.logwarn("无法移到观察位，尝试在当前位置拍照")
        rospy.sleep(1)

        # 获取最新图像
        image_array = self.camera.get_latest_image()
        if image_array is None:
            rospy.logerr("无法获取图像")
            return None

        rospy.loginfo("使用Qwen视觉模型分析图像...")
        board = self.vision.detect_board_state_from_array(image_array)

        if board:
            rospy.loginfo(f"检测到棋盘状态:\n{self.vision.format_board_display(board)}")
        else:
            rospy.logerr("棋盘检测失败")

        return board

    def get_human_move_manual(self):
        """
        通过手动输入获取人类玩家的移动

        Returns:
            list: 更新后的棋盘状态
        """
        while True:
            rospy.loginfo("\n" + "=" * 40)
            rospy.loginfo("人类回合 - 请放置您的棋子(X)")
            rospy.loginfo("=" * 40)

            print("\n当前棋盘状态:")
            print(self.logic.format_board_display())
            print("\n请输入您放置棋子的位置（行 列），或输入'q'退出:")

            try:
                user_input = raw_input("> ")  # Python 2
            except NameError:
                user_input = input("> ")  # Python 3

            if user_input.lower() == 'q':
                return None

            try:
                parts = user_input.strip().split()
                if len(parts) != 2:
                    raise ValueError("需要两个数字")

                row, col = int(parts[0]), int(parts[1])

                if row < 0 or row > 2 or col < 0 or col > 2:
                    raise ValueError("行列值必须在0-2之间")

                if self.board[row][col] != 0:
                    rospy.logerr(f"位置 ({row}, {col}) 已被占用！")
                    continue

                # 更新棋盘
                self.board[row][col] = 2  # 人类棋子
                rospy.loginfo(f"人类放置棋子到 ({row}, {col})")
                return self.board

            except ValueError as e:
                rospy.logerr(f"输入无效: {e}")
                rospy.loginfo("请重新输入（格式: 行 列，如: 0 1）")
                continue

    def get_human_move_auto(self):
        """
        通过视觉自动检测人类的移动

        Returns:
            list: 更新后的棋盘状态
        """
        rospy.loginfo("\n" + "=" * 40)
        rospy.loginfo("人类回合 - 视觉检测模式")
        rospy.loginfo("=" * 40)

        # 保存当前棋盘状态用于比较
        previous_board = [row[:] for row in self.board]

        rospy.loginfo("请人类放置棋子，然后按Enter继续...")
        try:
            raw_input()  # Python 2
        except NameError:
            input()  # Python 3

        # 捕获并分析图像
        current_board = self.capture_and_detect_board()

        if current_board is None:
            rospy.logerr("视觉检测失败，请使用手动输入")
            return self.get_human_move_manual()

        if not self.vision.validate_board_state(current_board):
            rospy.logerr("棋盘状态验证失败")
            return self.get_human_move_manual()

        # 比较找出人类的新棋子
        new_positions = []
        for i in range(3):
            for j in range(3):
                if previous_board[i][j] == 0 and current_board[i][j] == 2:
                    new_positions.append((i, j))

        if len(new_positions) == 0:
            rospy.logwarn("未检测到人类的新移动")
            rospy.loginfo("请确认您已经放置了棋子，或使用手动输入模式")
            return self.get_human_move_manual()
        elif len(new_positions) > 1:
            rospy.logwarn(f"检测到多个新棋子位置: {new_positions}，使用第一个")

        # 更新棋盘
        row, col = new_positions[0]
        self.board[row][col] = 2
        rospy.loginfo(f"检测到人类放置棋子到 ({row}, {col})")

        return self.board

    def robot_turn(self):
        """
        执行机器人回合

        真实场景流程：
        1. 从取棋区抓取深蓝色棋子
        2. 移动到棋盘指定位置
        3. 放置棋子

        Returns:
            bool: 是否成功
        """
        rospy.loginfo("\n" + "=" * 40)
        rospy.loginfo("机器人回合 - 思考中...")
        rospy.loginfo("=" * 40)

        # 让LLM生成最优移动
        move = self.logic.generate_move(self.board)

        if move is None:
            rospy.logerr("无法生成有效移动")
            return False

        row, col = move['row'], move['col']
        reason = move.get('reason', '')
        rospy.loginfo(f"机器人选择: ({row}, {col})")
        rospy.loginfo(f"决策原因: {reason}")

        # ========== 真实场景：从取棋区取棋子 ==========
        rospy.loginfo("=== 步骤1: 从取棋区抓取深蓝色棋子 ===")
        if not self.control.pick_piece_from_storage():
            rospy.logerr("从取棋区抓取棋子失败！")
            self.control.go_to_home()  # 回到安全位置
            return False

        # ========== 真实场景：放置棋子到棋盘 ==========
        rospy.loginfo(f"=== 步骤2: 移动到棋盘格子 ({row}, {col}) ===")
        if not self.control.place_piece(row, col):
            rospy.logerr("棋子放置失败！")
            self.control.open_gripper()
            rospy.sleep(0.5)
            self.control.go_to_home()  # 回到安全位置
            return False

        # 更新棋盘状态
        self.board[row][col] = 1  # 机器人棋子
        self.logic.update_board(self.board)

        rospy.loginfo("机器人回合完成")
        rospy.loginfo(f"\n当前棋盘状态:\n{self.logic.format_board_display()}")

        # 机械臂回安全位置
        self.control.go_to_home()

        return True

    def check_game_end(self):
        """
        检查游戏是否结束

        Returns:
            int: 0=继续, 1=机器人胜, 2=人类胜, 3=平局
        """
        winner = self.logic.check_winner(self.board)
        if winner == 1:
            return 1
        if winner == 2:
            return 2
        if self.logic.is_board_full(self.board):
            return 3
        return 0

    def initial_board_detection(self):
        """
        游戏开始前检测初始棋盘状态（如果有的话）

        Returns:
            bool: 是否检测成功
        """
        if self.camera is None or not self.camera.is_available():
            return False

        rospy.loginfo("检测初始棋盘状态...")
        board = self.capture_and_detect_board()

        if board and self.vision.validate_board_state(board):
            # 检查是否有非空的情况
            has_pieces = any(cell != 0 for row in board for cell in row)
            if has_pieces:
                rospy.loginfo("检测到已有棋子，使用当前状态作为初始棋盘")
                self.board = board
                self.logic.update_board(self.board)
                return True
        return False

    def shutdown(self):
        """关闭所有模块"""
        rospy.loginfo("关闭游戏系统...")
        if self.camera is not None:
            self.camera.shutdown()
        self.control.shutdown()
        rospy.loginfo("游戏系统已关闭")

    def game_loop(self):
        """游戏主循环"""
        rospy.loginfo("\n" + "=" * 50)
        rospy.loginfo("        井字棋游戏开始！")
        rospy.loginfo("=" * 50)
        rospy.loginfo("人类: X, 机器人: O")
        rospy.loginfo("机器人先手")
        rospy.loginfo("=" * 50)

        # 机械臂回初始位置
        rospy.loginfo("机械臂回初始位置...")
        self.control.go_to_home()
        self.control.open_gripper()

        # 可选：检测初始棋盘状态
        if self.auto_capture:
            self.initial_board_detection()

        rospy.loginfo(f"\n初始棋盘状态:\n{self.logic.format_board_display()}")

        try:
            turn = 0  # 0=机器人, 1=人类
            robot_fail_count = 0
            MAX_ROBOT_RETRIES = 3

            while not rospy.is_shutdown():
                # 检查游戏是否结束
                game_result = self.check_game_end()
                if game_result == 1:
                    rospy.loginfo("\n" + "=" * 50)
                    rospy.loginfo("  游戏结束！机器人获胜！")
                    rospy.loginfo("=" * 50)
                    break
                elif game_result == 2:
                    rospy.loginfo("\n" + "=" * 50)
                    rospy.loginfo("  游戏结束！人类获胜！")
                    rospy.loginfo("=" * 50)
                    break
                elif game_result == 3:
                    rospy.loginfo("\n" + "=" * 50)
                    rospy.loginfo("  游戏结束！平局！")
                    rospy.loginfo("=" * 50)
                    break

                if turn == 0:
                    # ========== 机器人回合 ==========
                    rospy.loginfo("\n" + "+" * 40)
                    rospy.loginfo("+        机器人回合")
                    rospy.loginfo("+" * 40)

                    if not self.robot_turn():
                        robot_fail_count += 1
                        rospy.logerr(f"机器人回合失败({robot_fail_count}/{MAX_ROBOT_RETRIES})")
                        if robot_fail_count >= MAX_ROBOT_RETRIES:
                            rospy.logerr("机器人回合连续失败次数过多，游戏结束")
                            break
                        rospy.sleep(1)
                        continue
                    else:
                        robot_fail_count = 0

                    turn = 1  # 切换到人类回合

                else:
                    # ========== 人类回合 ==========
                    rospy.loginfo("\n" + "+" * 40)
                    rospy.loginfo("+        人类回合")
                    rospy.loginfo("+" * 40)

                    if self.manual_human_move:
                        # 手动输入模式
                        updated_board = self.get_human_move_manual()
                        if updated_board is None:
                            rospy.loginfo("人类选择退出游戏")
                            break
                        self.board = updated_board
                        self.logic.update_board(self.board)
                    else:
                        # 自动检测模式
                        detected_board = self.get_human_move_auto()
                        if detected_board is None:
                            rospy.logerr("无法检测人类移动")
                            rospy.sleep(1)
                            continue
                        self.board = detected_board
                        self.logic.update_board(self.board)

                    rospy.loginfo(f"\n当前棋盘状态:\n{self.logic.format_board_display()}")

                    turn = 0  # 切换到机器人回合

                rospy.sleep(1)

        except KeyboardInterrupt:
            rospy.loginfo("\n游戏被用户中断 (Ctrl+C)")
        finally:
            rospy.loginfo("\n清理并关闭...")
            self.shutdown()
            rospy.loginfo("游戏结束")


def main():
    game = TicTacToeGame()
    game.game_loop()


if __name__ == "__main__":
    main()
