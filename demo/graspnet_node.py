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
import numpy as np

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'

# Endpoint del server di inferenza GraspNet, in ascolto sulla VM GPU
# (rocco-gpu-1). Richiede connessione VPN GlobalProtect attiva.
GRASP_SERVER_URL = "http://10.75.4.28:5001/predict_grasp"

# Soglie di profondita' (mm) per isolare la zona di lavoro. TODO: tarare
# empiricamente in lab — questi sono valori di partenza plausibili per
# la fase di grasp (braccio gia' vicino all'oggetto), non ancora validati.
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 400


from scipy.spatial.transform import Rotation
import base64
import requests

def rotation_matrix_to_quaternion(rotation_matrix, current_orientation_xyzw):
    """Converte la rotation_matrix 3x3 restituita da GraspNet (best_grasp.rotation_matrix,
    in camera frame) in un quaternione [x, y, z, w]. Include la correzione di segno per
    il percorso più breve rispetto all'orientamento corrente, stesso principio già usato
    in graspnetOrientationCallback (ET_node.cpp) e validato in
    test_standalone/test_rotation_to_quaternion.cpp.

    NON ANCORA CHIAMATA DA NESSUNA PARTE — pronta per quando l'inferenza vera
    sara' collegata (vedi TODO in trigger_callback)."""
    quat = Rotation.from_matrix(rotation_matrix).as_quat()

    dot = sum(quat[i] * current_orientation_xyzw[i] for i in range(4))
    if dot < 0.0:
        quat = -quat

    return quat


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

        self.get_logger().info('Trigger ricevuto: chiamo il server GraspNet sulla VM.')

        # --- 1. Conversione dei messaggi ROS in array numpy ---
        # cv_bridge usato qui (non rimosso come in sgg_ros_node.py) — se in
        # futuro emergesse lo stesso conflitto NumPy 1.x/2.x gia' visto
        # altrove, sostituire con conversione manuale come fatto la'.
        color_img = self.bridge.imgmsg_to_cv2(self.last_color_frame, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(self.last_depth_frame, desired_encoding='passthrough')
        depth_img = depth_img.astype('float32')

        # --- 2. Estrazione intrinseci dalla CameraInfo ---
        # msg.k e' una matrice 3x3 appiattita in row-major:
        # [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        k = self.last_camera_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]

        # --- 3. Chiamata HTTP al server GraspNet sulla VM ---
        def encode(array):
            return base64.b64encode(array.tobytes()).decode('utf-8')

        payload = {
            'color': encode(color_img), 'color_shape': list(color_img.shape), 'color_dtype': str(color_img.dtype),
            'depth': encode(depth_img), 'depth_shape': list(depth_img.shape), 'depth_dtype': str(depth_img.dtype),
            'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
            'depth_scale': 0.001,  # 1mm per unita' — confermato: e' il formato standard
                                    # Z16 di /camera/camera/aligned_depth_to_color/image_raw
                                    # (uint16, valori in mm), stesso topic e stessa assunzione
                                    # usati da imgmsg_to_numpy_depth16() in sgg_ros_node.py.
            'depth_min_mm': DEPTH_MIN_MM, 'depth_max_mm': DEPTH_MAX_MM,
        }

        try:
            resp = requests.post(GRASP_SERVER_URL, json=payload, timeout=10.0)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as exc:
            # Copre: VM irraggiungibile, VPN spenta, timeout, server giu'.
            self.get_logger().error(f'Chiamata al server GraspNet fallita: {exc}')
            return

        if not result.get('success'):
            self.get_logger().warn(f"GraspNet non ha prodotto un grasp valido: {result.get('reason')}")
            return

        # --- 4. Conversione rotation_matrix -> quaternione ---
        rotation_matrix = np.array(result['rotation_matrix'])
        # Orientamento corrente del gripper, come riferimento per la
        # correzione di segno (percorso piu' breve) dentro
        # rotation_matrix_to_quaternion(). Placeholder identita' — verificato
        # (12/09): NON sostituibile con un tf2 lookup diretto come in
        # moveit_goal_node.py (che usa tf2_ros.Buffer/TransformListener per
        # base_link->tool0). Qui serve un riferimento nello STESSO frame di
        # rotation_matrix, che la docstring di rotation_matrix_to_quaternion()
        # dichiara esplicitamente "in camera frame" — non base_link. tool0 in
        # base_link (il valore validato in moveit_goal_node.py) sarebbe quindi
        # un riferimento nel frame sbagliato: userebbe una correzione di segno
        # scorretta, peggio del placeholder identita' attuale (che almeno e'
        # un no-op trasparente). Nota collaterale emersa verificando questo:
        # graspnetOrientationCallback in ET_node.cpp riceve questo stesso
        # quaternione e lo usa direttamente come desired_ee_ori_acs_ (frame
        # base_link, nessuna trasformazione applicata) nonostante il messaggio
        # dichiari frame_id='camera_color_optical_frame' — possibile
        # disallineamento di convenzione tra le due repo, da chiarire con
        # Alessandro prima di toccare l'una o l'altra parte. TODO: per fixare
        # questo placeholder serve prima stabilire il vero frame di
        # rotation_matrix (chiedere a chi ha scritto grasp_server.py / la
        # repo graspnet-baseline) e, se serve davvero camera frame, non esiste
        # ancora una fonte per l'orientamento della camera in quel frame in
        # questa repo.
        current_orientation_xyzw = [0.0, 0.0, 0.0, 1.0]
        quat = rotation_matrix_to_quaternion(rotation_matrix, current_orientation_xyzw)

        # --- 5. Pubblicazione ---
        orientation_msg = QuaternionStamped()
        orientation_msg.header.stamp = self.get_clock().now().to_msg()
        orientation_msg.header.frame_id = 'camera_color_optical_frame'
        orientation_msg.quaternion.x = float(quat[0])
        orientation_msg.quaternion.y = float(quat[1])
        orientation_msg.quaternion.z = float(quat[2])
        orientation_msg.quaternion.w = float(quat[3])

        self.pub.publish(orientation_msg)
        self.get_logger().info(f"Grasp pubblicato (score={result['score']:.3f}).")


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