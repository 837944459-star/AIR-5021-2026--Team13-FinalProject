#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tic_tac_toe_logic.py
井字棋游戏逻辑和LLM策略生成
"""

import json
import os
import re
import requests
import rospy
from std_msgs.msg import String


class TicTacToeLogic:
    def __init__(self, api_key=None, model="qwen-plus"):
        """
        初始化游戏逻辑模块

        Args:
            api_key: Qwen API密钥
            model: 使用的LLM模型
        """
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供Qwen API密钥！设置环境变量 QWEN_API_KEY 或传入api_key参数")

        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

        # 棋盘状态
        self.EMPTY, self.ROBOT, self.HUMAN = 0, 1, 2
        self.board = [[0] * 3 for _ in range(3)]

    def check_winner(self, board=None):
        """
        检查获胜者

        Args:
            board: 3x3棋盘状态，如果为None则使用内部状态

        Returns:
            int: 0=无胜者, 1=机器人胜, 2=人类胜
        """
        if board is None:
            board = self.board

        # 行检查
        for row in board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]

        # 列检查
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col] != 0:
                return board[0][col]

        # 对角线检查
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]

        return 0

    def is_board_full(self, board=None):
        """
        检查棋盘是否已满

        Args:
            board: 3x3棋盘状态，如果为None则使用内部状态

        Returns:
            bool: 棋盘是否已满
        """
        if board is None:
            board = self.board

        for row in board:
            for cell in row:
                if cell == 0:
                    return False
        return True

    def get_empty_positions(self, board=None):
        """
        获取所有空位

        Args:
            board: 3x3棋盘状态，如果为None则使用内部状态

        Returns:
            list: 空位坐标列表 [(row, col), ...]
        """
        if board is None:
            board = self.board

        positions = []
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    positions.append((i, j))
        return positions

    def update_board(self, board):
        """
        更新内部棋盘状态

        Args:
            board: 3x3棋盘状态
        """
        self.board = board

    def generate_move(self, board=None):
        """
        使用LLM生成下一步移动

        Args:
            board: 3x3当前棋盘状态，如果为None则使用内部状态

        Returns:
            dict: {"row": int, "col": int, "reason": str} 或 None
        """
        if board is None:
            board = self.board

        # 检查是否还有空位
        empty = self.get_empty_positions(board)
        if not empty:
            rospy.logwarn("棋盘已满，无法生成移动")
            return None

        # 构建提示
        board_str = self._format_board(board)

        prompt = f"""你是一个井字棋(Tic-Tac-Toe)游戏策略专家。

**真实场景说明：**
- 棋盘：15cm×15cm硬纸板，黑色水笔绘制网格，每格5cm×5cm
- 机器人棋子：深蓝色小方块，棱长约2.4cm
- 人类棋子：黄色小方块，棱长约2.4cm
- 机器人先手

当前棋盘状态（0=空，1=机器人深蓝色方块，2=人类黄色方块）：
{board_str}

你是机器人（深蓝色方块O），请根据当前局势选择最优的下一步移动。
你需要：
1. 首先检查是否有机会赢（机器人深蓝色O能连成一条线）
2. 如果不能赢，检查是否需要阻挡人类（黄色X）的胜利
3. 如果都不需要，选择一个战略位置（中心或角落）

请以JSON格式返回你的决策：
{{"row": 行号(0-2), "col": 列号(0-2), "reason": "决策原因"}}

只返回JSON，不要有其他文字。"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": {"prompt": prompt},
            "parameters": {"temperature": 0.7, "max_tokens": 200}
        }

        rospy.loginfo("调用Qwen LLM生成策略...")

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                rospy.logerr(f"LLM API调用失败: {response.status_code}")
                rospy.logerr(f"响应内容: {response.text}")
                # Fallback: 返回第一个空位
                return self._get_fallback_move(empty)

            result = response.json()
            content = result['output']['text']
            rospy.logdebug(f"LLM返回内容: {content}")

            # 解析JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                move_data = json.loads(json_match.group())
                row = int(move_data.get('row', -1))
                col = int(move_data.get('col', -1))

                # 验证移动合法性
                if 0 <= row <= 2 and 0 <= col <= 2 and board[row][col] == 0:
                    rospy.loginfo(f"LLM决策: ({row}, {col}) - {move_data.get('reason', '')}")
                    return {"row": row, "col": col, "reason": move_data.get('reason', '')}
                else:
                    rospy.logwarn(f"LLM返回了非法移动: row={row}, col={col}，使用fallback")
                    return self._get_fallback_move(empty)
            else:
                rospy.logwarn("无法从LLM响应中提取JSON，使用fallback")
                return self._get_fallback_move(empty)

        except Exception as e:
            rospy.logerr(f"生成移动失败: {e}")
            return self._get_fallback_move(empty)

    def _get_fallback_move(self, empty_positions):
        """
        获取fallback移动（第一个空位）

        Args:
            empty_positions: 空位列表

        Returns:
            dict: 第一个空位的移动
        """
        if empty_positions:
            row, col = empty_positions[0]
            rospy.logwarn(f"Fallback: 选择第一个空位 ({row}, {col})")
            return {"row": row, "col": col, "reason": "Fallback: first empty cell"}
        return None

    def _format_board(self, board):
        """
        格式化棋盘为可读字符串

        Args:
            board: 3x3棋盘状态

        Returns:
            str: 格式化后的棋盘
        """
        symbols = {0: '.', 1: 'O', 2: 'X'}
        result = []
        for i, row in enumerate(board):
            result.append(f"Row {i}: " + " ".join(symbols[c] for c in row))
        return "\n".join(result)

    def format_board_display(self, board=None):
        """
        格式化棋盘为可读字符串用于显示

        Args:
            board: 3x3棋盘状态，如果为None则使用内部状态

        Returns:
            str: 格式化后的棋盘字符串
        """
        if board is None:
            board = self.board

        symbols = {0: '.', 1: 'O', 2: 'X'}
        lines = []
        for i, row in enumerate(board):
            lines.append(f"  {i} | " + " | ".join(symbols[c] for c in row))
            if i < 2:
                lines.append("  --+---+--")
        return "\n    0   1   2\n" + "\n".join(lines)


def main():
    rospy.init_node('tic_tac_toe_logic_test')

    api_key = rospy.get_param('~api_key', None)

    rospy.loginfo("初始化游戏逻辑模块...")
    logic = TicTacToeLogic(api_key=api_key)

    # 测试棋盘
    test_board = [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]

    rospy.loginfo(f"测试棋盘:\n{logic.format_board_display(test_board)}")

    move = logic.generate_move(test_board)
    if move:
        rospy.loginfo(f"生成移动: {move}")
    else:
        rospy.logerr("无法生成移动")


if __name__ == "__main__":
    main()
