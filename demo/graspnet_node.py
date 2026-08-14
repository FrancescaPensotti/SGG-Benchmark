#!/usr/bin/env python3
"""
Nodo placeholder per l'integrazione GraspNet (Fase B).

Ascolta /graspnet/trigger (pubblicato da ET_node quando il braccio entra in
zona di grasp) e risponde pubblicando un orientamento su
/graspnet/grasp_orientation.

Per ora l'orientamento è un PLACEHOLDER fisso, non calcolato da GraspNet
(bloccato dalla mancanza di GPU sul laptop). Questo nodo
serve a validare l'intera tubatura (trigger -> risposta -> movimento del
braccio) prima che l'inferenza vera sia disponibile. Quando GraspNet sarà
collegabile, questo file diventerà lo scheletro su cui innestare la vera
cattura RGB-D + inferenza, sostituendo solo la funzione trigger_callback.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import QuaternionStamped


class GraspNetNode(Node):
    def __init__(self):
        super().__init__('graspnet_node')

        self.declare_parameter('trigger_topic', '/graspnet/trigger')
        self.declare_parameter('orientation_topic', '/graspnet/grasp_orientation')

        trigger_topic = self.get_parameter('trigger_topic').get_parameter_value().string_value
        orientation_topic = self.get_parameter('orientation_topic').get_parameter_value().string_value

        self.pub = self.create_publisher(QuaternionStamped, orientation_topic, 10)
        self.sub = self.create_subscription(
            Bool, trigger_topic, self.trigger_callback, 10
        )

        self.get_logger().info(
            f'GraspNet node (placeholder) attivo: {trigger_topic} -> {orientation_topic}'
        )

    def trigger_callback(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().info('Trigger ricevuto: pubblico orientamento placeholder.')

        # PLACEHOLDER: quaternione ruotato di 90° attorno a Z rispetto
        # all'identità, scelto apposta diverso da (0,0,0,1) per verificare
        # visivamente che il braccio reagisca al messaggio.
        orientation_msg = QuaternionStamped()
        orientation_msg.header.stamp = self.get_clock().now().to_msg()
        orientation_msg.header.frame_id = 'base_link'
        orientation_msg.quaternion.x = 0.0
        orientation_msg.quaternion.y = 0.0
        orientation_msg.quaternion.z = 0.7071
        orientation_msg.quaternion.w = 0.7071

        self.pub.publish(orientation_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()