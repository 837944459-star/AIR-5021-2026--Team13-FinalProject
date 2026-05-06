#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_capture.py
摄像头图像获取模块
从ROS摄像头节点获取图像并保存
"""

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os
import time


class CameraCapture:
    def __init__(self, image_topic=None):
        """
        初始化摄像头图像获取

        Args:
            image_topic: 图像话题名称，如果为None则从参数服务器读取
        """
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_received = False

        # 获取图像话题名称
        if image_topic is None:
            # 尝试从参数服务器读取，默认使用常见的摄像头topic
            self.image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        else:
            self.image_topic = image_topic

        rospy.loginfo(f"订阅图像话题: {self.image_topic}")

        # 订阅图像话题
        self.subscriber = rospy.Subscriber(
            self.image_topic,
            Image,
            self._image_callback,
            queue_size=1
        )

        # 等待图像接收
        timeout = 10  # 超时时间（秒）
        start_time = rospy.Time.now()
        rospy.loginfo("等待摄像头图像...")

        while not self.image_received and rospy.Time.now() - start_time < rospy.Duration(timeout):
            rospy.sleep(0.1)

        if not self.image_received:
            rospy.logwarn(f"在 {timeout} 秒内未收到图像，请检查摄像头是否启动")

    def _image_callback(self, data):
        """图像回调函数"""
        try:
            # 将ROS图像转换为OpenCV图像
            self.latest_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            self.image_received = True
        except CvBridgeError as e:
            rospy.logerr(f"图像转换失败: {e}")

    def capture_image(self, save_path=None):
        """
        捕获当前图像

        Args:
            save_path: 保存路径，如果为None则使用临时目录

        Returns:
            str: 保存的图像路径，或None如果捕获失败
        """
        if self.latest_image is None:
            rospy.logerr("没有可用的图像")
            return None

        # 如果没有指定保存路径，使用临时目录
        if save_path is None:
            save_dir = rospy.get_param('~save_dir', '/tmp')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f'board_capture_{timestamp}.jpg')

        try:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存图像
            cv2.imwrite(save_path, self.latest_image)
            rospy.loginfo(f"图像已保存: {save_path}")
            return save_path

        except Exception as e:
            rospy.logerr(f"图像保存失败: {e}")
            return None

    def capture_and_wait(self, save_path=None, timeout=5.0):
        """
        等待并捕获图像（用于初始化时）

        Args:
            save_path: 保存路径
            timeout: 超时时间

        Returns:
            str: 保存的图像路径，或None
        """
        start_time = rospy.Time.now()
        rate = rospy.Rate(10)

        while not self.image_received and rospy.Time.now() - start_time < rospy.Duration(timeout):
            rate.sleep()

        if self.image_received:
            return self.capture_image(save_path)
        else:
            rospy.logerr("等待图像超时")
            return None

    def get_latest_image(self):
        """
        获取最新图像（不保存）

        Returns:
            numpy.ndarray: OpenCV图像，或None
        """
        return self.latest_image

    def is_available(self):
        """
        检查摄像头是否可用

        Returns:
            bool: 摄像头是否已接收图像
        """
        return self.image_received

    def shutdown(self):
        """关闭订阅"""
        if hasattr(self, 'subscriber'):
            self.subscriber.unregister()
            rospy.loginfo("摄像头订阅已关闭")


def main():
    rospy.init_node('camera_capture_test')

    # 获取参数
    image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
    save_path = rospy.get_param('~save_path', '/tmp/test_capture.jpg')

    rospy.loginfo(f"测试摄像头捕获，topic: {image_topic}")

    # 初始化摄像头捕获
    capture = CameraCapture(image_topic=image_topic)

    if capture.is_available():
        rospy.loginfo("摄像头初始化成功")

        # 捕获图像
        saved_path = capture.capture_image(save_path)
        if saved_path:
            rospy.loginfo(f"图像已保存到: {saved_path}")
        else:
            rospy.logerr("图像捕获失败")
    else:
        rospy.logerr("摄像头初始化失败，请检查：")
        rospy.logerr("1. 摄像头是否连接")
        rospy.logerr("2. 图像话题名称是否正确")
        rospy.logerr("3. 是否有其他节点在发布图像")

    capture.shutdown()


if __name__ == "__main__":
    main()
