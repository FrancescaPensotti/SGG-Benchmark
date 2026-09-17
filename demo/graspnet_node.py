#!/usr/bin/env python3
"""
Nodo di integrazione GraspNet.

Ascolta /graspnet/trigger (pubblicato da ET_node quando il braccio entra in
zona di grasp): cattura l'ultimo frame RGB-D disponibile, lo manda al
server di inferenza GraspNet sulla VM GPU (grasp_server.py, via HTTP),
converte la posa di grasp restituita in un quaternione e lo pubblica su
/graspnet/grasp_orientation.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import QuaternionStamped, PoseArray, PointStamped
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


def imgmsg_to_numpy_bgr8(msg):
    """Sostituisce cv_bridge.imgmsg_to_cv2(msg, 'bgr8') con una conversione
    manuale — stesso motivo di sgg_ros_node.py: cv_bridge (compilato contro
    NumPy 1.x dal sistema ROS2) va in segfault se importato insieme a
    NumPy 2.x (qui richiesto da altre dipendenze del venv)."""
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)


def imgmsg_to_numpy_depth16(msg):
    """Sostituisce cv_bridge.imgmsg_to_cv2(msg, 'passthrough') per la depth
    16UC1 (RealSense allineata, valori in millimetri) — stesso motivo sopra."""
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
CAMERA_INFO_TOPIC = '/camera/camera/color/camera_info'

# Endpoint del server di inferenza GraspNet, in ascolto sulla VM GPU
# (rocco-gpu-1). Richiede connessione VPN GlobalProtect attiva.
GRASP_SERVER_URL = "http://10.75.4.28:5001/predict_grasp"

# Soglie di profondita' (mm) per isolare la zona di lavoro. TODO: tarare
# empiricamente in lab — questi sono valori di partenza plausibili per
# la fase di grasp (braccio gia' vicino all'oggetto), non ancora validati.
# Al trigger (zona di grasp a 0.30 m dal punto di hover) la camera e' a circa
# 0.4 m dall'oggetto: con un massimo di 400 mm l'oggetto veniva tagliato.
DEPTH_MIN_MM = 150
DEPTH_MAX_MM = 700

# Raggio entro cui un grasp e' considerato "sul target" (camera frame, metri).
# Tiene conto della latenza tra l'immagine di SGG e quella usata per GraspNet.
GRASP_TARGET_RADIUS_M = 0.10


from scipy.spatial.transform import Rotation
import base64
import requests

def rotation_matrix_to_quaternion(rotation_matrix, current_orientation_xyzw):
    """Converte la rotation_matrix 3x3 restituita da GraspNet (best_grasp.rotation_matrix,
    in camera frame) in un quaternione [x, y, z, w]. Include la correzione di segno per
    il percorso più breve rispetto all'orientamento corrente, stesso principio già usato
    in graspnetOrientationCallback (ET_node.cpp) e validato in
    test_standalone/test_rotation_to_quaternion.cpp.

    Chiamata da trigger_callback dopo la risposta del server GraspNet (vedi
    sopra in questo file)."""
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
        self.declare_parameter('candidate_targets_topic', '/sgg/candidate_targets')
        self.declare_parameter('target_point_topic', '/sgg/target_point')

        trigger_topic = self.get_parameter('trigger_topic').get_parameter_value().string_value
        orientation_topic = self.get_parameter('orientation_topic').get_parameter_value().string_value

        self.pub = self.create_publisher(QuaternionStamped, orientation_topic, 10)
        # Coda di 1: la chiamata al server blocca il thread, e i trigger
        # accumulati nel frattempo produrrebbero risposte su frame vecchi.
        self.sub = self.create_subscription(
            Bool, trigger_topic, self.trigger_callback, 1
        )

        # Ultima posizione nota del target in camera frame, dal canale che ha
        # pubblicato per ultimo (candidati impliciti o target esplicito t/g).
        self.last_target_points = []
        self.create_subscription(
            PoseArray, self.get_parameter('candidate_targets_topic').get_parameter_value().string_value,
            self.candidates_callback, 10
        )
        self.create_subscription(
            PointStamped, self.get_parameter('target_point_topic').get_parameter_value().string_value,
            self.target_point_callback, 10
        )

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

    def candidates_callback(self, msg: PoseArray):
        self.last_target_points = [np.array([p.position.x, p.position.y, p.position.z]) for p in msg.poses]

    def target_point_callback(self, msg: PointStamped):
        self.last_target_points = [np.array([msg.point.x, msg.point.y, msg.point.z])]

    def select_grasp(self, result):
        """Il grasp con score piu' alto entro GRASP_TARGET_RADIUS_M da un
        target noto (i grasp arrivano gia' ordinati per score). Senza target
        noti usa il migliore in assoluto. None se nessun grasp e' sul target."""
        grasps = result.get('grasps') or [result]
        if not self.last_target_points:
            self.get_logger().warn('Nessun target noto: uso il grasp migliore in assoluto.')
            return grasps[0]
        for g in grasps:
            t = np.array(g['translation'])
            dist = min(np.linalg.norm(t - p) for p in self.last_target_points)
            if dist < GRASP_TARGET_RADIUS_M:
                self.get_logger().info(f"Grasp sul target: distanza {dist:.3f} m, score {g['score']:.3f}.")
                return g
        best_dist = min(min(np.linalg.norm(np.array(g['translation']) - p) for p in self.last_target_points) for g in grasps)
        self.get_logger().warn(
            f'Nessuno dei {len(grasps)} grasp entro {GRASP_TARGET_RADIUS_M} m dal target '
            f'(il piu\' vicino a {best_dist:.3f} m): nessun orientamento pubblicato.')
        return None

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
        # Conversione manuale (non cv_bridge, vedi imgmsg_to_numpy_bgr8/
        # imgmsg_to_numpy_depth16 in cima al file) -- cv_bridge qui andava in
        # segfault al primo trigger reale (conflitto NumPy 1.x/2.x), stesso
        # problema gia' risolto in sgg_ros_node.py.
        color_img = imgmsg_to_numpy_bgr8(self.last_color_frame)
        depth_img = imgmsg_to_numpy_depth16(self.last_depth_frame)
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

        grasp = self.select_grasp(result)
        if grasp is None:
            return

        # --- 4. Conversione rotation_matrix -> quaternione ---
        rotation_matrix = np.array(grasp['rotation_matrix'])
        # Orientamento corrente del gripper, come riferimento per la
        # correzione di segno (percorso piu' breve) dentro
        # rotation_matrix_to_quaternion(). Placeholder identita'.
        #
        # graspnetOrientationCallback (ET_node.cpp) compone gia' correttamente
        # ee_orientation_ * camera_to_tool0_ * q_camera_frame (stessa
        # convenzione "camera frame" usata qui), con test di autoconsistenza
        # algebrica in test_standalone/test_camera_to_baselink_orientation.cpp
        # -- nessun problema di frame da quella parte.
        #
        # Il placeholder resta per un motivo diverso e minore:
        # rotation_matrix_to_quaternion() cerca il percorso piu' breve
        # rispetto a un orientamento "corrente" per evitare un salto tra due
        # rappresentazioni equivalenti del quaternione (doppio ricoprimento) --
        # qui non c'e' uno stato persistente tra una chiamata e la successiva
        # (trigger_callback parte sempre da zero) con cui confrontare, quindi
        # l'identita' resta la scelta piu' semplice finche' non si decide se
        # vale la pena mantenere l'ultimo orientamento pubblicato come
        # riferimento tra una chiamata e l'altra.
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
        self.get_logger().info(f"Grasp pubblicato (score={grasp['score']:.3f}).")


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