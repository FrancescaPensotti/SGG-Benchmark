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
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'


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

        self.bridge = CvBridge()

        # Ultimo frame disponibile per ciascuna sorgente — aggiornati in
        # continuo dalle rispettive callback, letti (non richiesti on-demand)
        # quando arriva il trigger. Nessun lock: le callback di questo nodo
        # girano tutte sul thread di default, nessuna concorrenza reale.
        self.last_color_frame = None
        self.last_depth_frame = None
        self.last_camera_info = None

        self.color_sub = self.create_subscription(
            Image, COLOR_TOPIC, self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10
        )


        self.get_logger().info(
            f'GraspNet node attivo: {trigger_topic} -> {orientation_topic} | '
            f'RGB-D da {COLOR_TOPIC}, {DEPTH_TOPIC}, {CAMERA_INFO_TOPIC}'
        )

    def color_callback(self, msg: Image):
        self.last_color_frame = msg

    def depth_callback(self, msg: Image):
        self.last_depth_frame = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info = msg

    def trigger_callback(self, msg: Bool):
        if not msg.data:
            return

        if self.last_color_frame is None or self.last_depth_frame is None or self.last_camera_info is None:
            self.get_logger().warn(
                    'Trigger ricevuto ma RGB-D non ancora disponibile '
                    '(color/depth/camera_info mancanti) — nessuna risposta pubblicata.'
                )
            return

        self.get_logger().info('Trigger ricevuto: RGB-D disponibile, pubblico orientamento placeholder.')

        # TODO: qui va innestata l'inferenza vera di GraspNet, quando avremo
        # accesso alla GPU (VM del Politecnico o PC del tutor). Passi previsti:
        #   1. Convertire self.last_color_frame e self.last_depth_frame da
        #      sensor_msgs/Image a array numpy (self.bridge.imgmsg_to_cv2).
        #   2. Estrarre la matrice degli intrinseci da self.last_camera_info.k.
        #   3. Eseguire l'inferenza GraspNet (repo in ~/tesi/graspnet-baseline)
        #      per ottenere la posa di grasp proposta.
        #   4. Convertire gli assi in quaternione (stessa logica già validata
        #      in test_standalone/test_rotation_to_quaternion.cpp).
        #   5. Pubblicare il quaternione risultante al posto del placeholder
        #      sotto (già in camera frame — la trasformazione a base_link la
        #      fa ET_node.cpp in graspnetOrientationCallback).

        # PLACEHOLDER: quaternione ruotato di 90° attorno a Z rispetto
        # all'identità, scelto apposta diverso da (0,0,0,1) per verificare
        # visivamente che il braccio reagisca al messaggio.
        orientation_msg = QuaternionStamped()
        orientation_msg.header.stamp = self.get_clock().now().to_msg()
        orientation_msg.header.frame_id = 'camera_color_optical_frame'
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