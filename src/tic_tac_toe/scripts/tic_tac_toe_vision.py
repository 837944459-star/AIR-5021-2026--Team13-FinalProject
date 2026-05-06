#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tic_tac_toe_vision.py
使用Qwen视觉模型检测井字棋棋盘状态
"""

import base64
import cv2
import json
import numpy as np
import os
import re
import requests
import rospy
from std_msgs.msg import String


class TicTacToeVision:
    def __init__(self, api_key=None, model="qwen-vl-max"):
        """
        初始化视觉模块

        Args:
            api_key: Qwen API密钥，如果为None则从环境变量QWEN_API_KEY读取
            model: 使用的Qwen模型，默认为qwen-vl-max
        """
        self.api_key = api_key or os.environ.get("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供Qwen API密钥！设置环境变量 QWEN_API_KEY 或传入api_key参数")

        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

        # 状态定义：0=空，1=机器人(O)，2=人类(X)
        self.EMPTY, self.ROBOT, self.HUMAN = 0, 1, 2

    def encode_image_to_base64(self, image_path):
        """将图像文件编码为base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def encode_image_to_base64_from_array(self, image_array):
        """
        将OpenCV图像数组编码为base64

        Args:
            image_array: numpy.ndarray, OpenCV图像 (BGR格式)

        Returns:
            str: base64编码的图像字符串
        """
        # OpenCV图像是BGR格式，cv2.imencode期望BGR格式
        # 编码为JPEG（imencode期望BGR格式）
        _, buffer = cv2.imencode('.jpg', image_array)
        return base64.b64encode(buffer).decode('utf-8')

    def detect_board_state(self, image_path):
        """
        使用Qwen视觉模型分析棋盘状态

        Args:
            image_path: 棋盘图像路径

        Returns:
            list: 3x3网格状态，如 [[0,0,0],[0,1,0],[0,0,2]]
        """
        prompt = """你是一个井字棋游戏的视觉分析助手。
请分析这张井字棋(Tic-Tac-Toe)棋盘图片。

**真实场景描述：**
- 棋盘：用黑色水笔在硬纸板上绘制的正方形，15cm×15cm，每格5cm×5cm
- 人类棋子：黄色小方块，棱长约2.4cm
- 机器人棋子：深蓝色小方块，棱长约2.4cm
- 棋盘格子用黑色水笔绘制，有明显的网格线

请返回JSON格式的棋盘状态：
- 0 表示该位置为空（没有棋子）
- 1 表示该位置有深蓝色方块（机器人O的棋子）
- 2 表示该位置有黄色方块（人类X的棋子）

请只返回JSON，不要有其他文字：
{"board": [[0,0,0],[0,0,0],[0,0,0]]}

例如，如果中心是机器人（深蓝色），左下是人类（黄色），则返回：
{"board": [[0,0,0],[0,1,0],[2,0,0]]}"""

        image_base64 = self.encode_image_to_base64(image_path)

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{image_base64}"},
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        rospy.loginfo(f"调用Qwen视觉API分析图像: {image_path}")

        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            rospy.logerr(f"Qwen API调用失败: {response.status_code}, {response.text}")
            return None

        result = response.json()

        # 解析Qwen返回的内容
        try:
            content = result['output']['choices'][0]['message']['content']
            if isinstance(content, list):
                content = content[0].get('text', '')
            rospy.logdebug(f"Qwen返回内容: {content}")

            # 提取JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                board_data = json.loads(json_match.group())
                return board_data.get('board')
        except (KeyError, json.JSONDecodeError) as e:
            rospy.logerr(f"解析Qwen返回失败: {e}")

        return None

    def detect_board_state_from_array(self, image_array):
        """
        使用Qwen视觉模型分析OpenCV图像数组

        Args:
            image_array: numpy.ndarray, OpenCV图像 (BGR格式)

        Returns:
            list: 3x3网格状态，如 [[0,0,0],[0,1,0],[0,0,2]]
        """
        prompt = """你是一个井字棋游戏的视觉分析助手。
请分析这张井字棋(Tic-Tac-Toe)棋盘图片。

**真实场景描述：**
- 棋盘：用黑色水笔在硬纸板上绘制的正方形，15cm×15cm，每格5cm×5cm
- 人类棋子：黄色小方块，棱长约2.4cm
- 机器人棋子：深蓝色小方块，棱长约2.4cm
- 棋盘格子用黑色水笔绘制，有明显的网格线

请返回JSON格式的棋盘状态：
- 0 表示该位置为空（没有棋子）
- 1 表示该位置有深蓝色方块（机器人O的棋子）
- 2 表示该位置有黄色方块（人类X的棋子）

请只返回JSON，不要有其他文字：
{"board": [[0,0,0],[0,0,0],[0,0,0]]}

例如，如果中心是机器人（深蓝色），左下是人类（黄色），则返回：
{"board": [[0,0,0],[0,1,0],[2,0,0]]}"""

        image_base64 = self.encode_image_to_base64_from_array(image_array)

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{image_base64}"},
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters":{
                "result_format": "message"
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        rospy.loginfo("调用Qwen视觉API分析图像（直接传入）")

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)

            if response.status_code != 200:
                rospy.logerr(f"Qwen API调用失败: {response.status_code}, {response.text}")
                return None

            result = response.json()

            # 解析Qwen返回的内容
            content = result['output']['choices'][0]['message']['content']
            if isinstance(content, list):
                content = content[0].get('text', '')
            rospy.logdebug(f"Qwen返回内容: {content}")

            # 提取JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                board_data = json.loads(json_match.group())
                return board_data.get('board')

        except Exception as e:
            rospy.logerr(f"检测棋盘状态失败: {e}")

        return None

    def validate_board_state(self, board):
        """
        验证棋盘状态的合法性

        Args:
            board: 3x3网格状态

        Returns:
            bool: 是否合法
        """
        if not board or len(board) != 3:
            return False
        for row in board:
            if len(row) != 3:
                return False
            for cell in row:
                if cell not in [0, 1, 2]:
                    return False
        return True

    def format_board_display(self, board):
        """
        格式化棋盘为可读字符串用于显示

        Args:
            board: 3x3网格状态

        Returns:
            str: 格式化后的棋盘字符串
        """
        symbols = {0: '.', 1: '深蓝(O)', 2: '黄(X)'}
        lines = []
        for i, row in enumerate(board):
            lines.append(f"  {i} | " + " | ".join(symbols[c] for c in row))
            if i < 2:
                lines.append("  --+---+--")
        return "\n    0   1   2\n" + "\n".join(lines)


def main():
    rospy.init_node('tic_tac_toe_vision_test')

    # 从参数获取测试图像路径
    test_image = rospy.get_param('~test_image', '/tmp/board.jpg')
    api_key = rospy.get_param('~api_key', None)

    rospy.loginfo("初始化视觉模块...")
    vision = TicTacToeVision(api_key=api_key)

    rospy.loginfo(f"分析图像: {test_image}")
    board = vision.detect_board_state(test_image)

    if board:
        rospy.loginfo(f"检测到棋盘状态:\n{vision.format_board_display(board)}")

        if vision.validate_board_state(board):
            rospy.loginfo("棋盘状态验证通过")
        else:
            rospy.logerr("棋盘状态验证失败")
    else:
        rospy.logerr("棋盘状态检测失败")


if __name__ == "__main__":
    main()
