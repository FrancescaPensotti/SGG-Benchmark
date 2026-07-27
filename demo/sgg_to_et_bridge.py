#!/usr/bin/env python3
"""
Nodo ponte: converte il target pubblicato da sgg_ros_node.py (PointStamped su
/sgg/target_point, in camera_color_optical_frame) in un PoseStamped sul topic
aruco_pose_topic, consumato da arucoPoseCallback in ET_node.cpp.

L'orientamento non viene attualmente usato da ET_node (la logica di orientamento
è controllata dall'utente, vedi Orientation Handling in arucoPoseCallback), quindi
qui pubblichiamo un quaternione identità come placeholder valido.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped


class SggToEtBridge(Node):
    def __init__(self):
        super().__init__('sgg_to_et_bridge')

        self.declare_parameter('input_topic', '/sgg/target_point')
        self.declare_parameter('output_topic', '/aruco_detector/target_pose_camera_frame')

        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.pub = self.create_publisher(PoseStamped, output_topic, 10)
        self.sub = self.create_subscription(
            PointStamped, input_topic, self.target_point_callback, 10
        )

        self.get_logger().info(
            f'Bridge attivo: {input_topic} (PointStamped) -> {output_topic} (PoseStamped)'
        )

    def target_point_callback(self, msg: PointStamped):
        pose_msg = PoseStamped()
        pose_msg.header = msg.header  # stesso frame_id e stamp del punto ricevuto

        pose_msg.pose.position.x = msg.point.x
        pose_msg.pose.position.y = msg.point.y
        pose_msg.pose.position.z = msg.point.z

        # Orientamento placeholder: non usato da ET_node nella logica attuale
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SggToEtBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()