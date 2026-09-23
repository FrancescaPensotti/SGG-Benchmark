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
from std_msgs.msg import Bool, Float32MultiArray
from geometry_msgs.msg import QuaternionStamped, PoseArray, PointStamped
from sensor_msgs.msg import Image, CameraInfo
import numpy as np


def imgmsg_to_numpy_bgr8(msg):
    """Sostituisce cv_bridge.imgmsg_to_cv2(msg, 'bgr8') con una conversione
    manuale — stesso motivo di sgg_ros_node.py: cv_bridge (compilato contro
    NumPy 1.x dal sistema ROS2) va in segfault se importato insieme a
    NumPy 2.x (qui richiesto da altre dipendenze del venv).

    Il topic REALE (/camera/camera/color/image_raw) pubblica in encoding
    'rgb8' (verificato dal vivo il 21/09/2026), non 'bgr8': questa funzione fa
    un reshape puro senza scambiare i canali. Un tentativo di correggere lo
    scambio (21-22/09) e' stato ripristinato: vedi imgmsg_to_numpy_bgr8() in
    sgg_ros_node.py per il motivo (ha peggiorato l'identificazione nei test
    reali, nonostante fosse corretto sulla carta). Qui l'immagine va solo al
    server GraspNet per colorare la point cloud, non per la geometria della
    presa -- tenuta coerente con sgg_ros_node.py comunque."""
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
GRASP_TARGET_RADIUS_M = 0.15

# Margine (metri, camera frame) aggiunto intorno alla bbox del target prima
# di mandarla a grasp_server.py per il crop X/Y della point cloud (22/09/2026).
# Deve coprire almeno GRASP_TARGET_RADIUS_M (altrimenti scarteremmo nel crop
# prese che poi il filtro di distanza accetterebbe comunque), piu' un
# margine extra perche' il collision detector abbia contesto intorno al
# bordo. Valore di partenza, da tarare in laboratorio come depth_min/max_mm.
BBOX_MARGIN_M = GRASP_TARGET_RADIUS_M + 0.05


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
        self.declare_parameter('target_bbox_topic', '/sgg/target_bbox')
        self.declare_parameter('bbox_margin_m', BBOX_MARGIN_M)
        self.bbox_margin_m = self.get_parameter('bbox_margin_m').get_parameter_value().double_value
        # Apertura massima utilizzabile: la Hand-E apre al massimo 5 cm, con
        # 5 mm di margine per l'errore di stima della larghezza. Valore
        # fisico reale -- riportato qui a fine sessione del 21/09/2026 dopo
        # gli alzamenti temporanei a 0.10 per i test diagnostici di oggi.
        self.declare_parameter('max_grasp_width', 0.045)
        self.max_grasp_width = self.get_parameter('max_grasp_width').get_parameter_value().double_value

        # Soglia minima di punteggio GraspNet: tra le prese sopra questa
        # soglia si sceglie quella che farebbe ruotare meno il polso rispetto
        # alla sua posa attuale (vedi select_grasp/_best_equivalent_rotation),
        # non piu' semplicemente la migliore per punteggio. Aggiunto il
        # 21/09/2026 su proposta del tutor: evita di scegliere una presa di
        # qualita' scarsa solo perche' richiede poca rotazione.
        self.declare_parameter('min_grasp_score', 0.5)
        self.min_grasp_score = self.get_parameter('min_grasp_score').get_parameter_value().double_value

        # Costanti fisse di calibrazione, stesse usate in ET_node.cpp
        # (camera_to_tool0_quat, grasp_frame_to_tool0_quat) -- servono SOLO a
        # stimare quanto dovrebbe ruotare il polso per ciascuna presa
        # candidata (vedi _best_equivalent_rotation): non rifanno la
        # trasformazione finale in base_link, che resta interamente in
        # ET_node (unica fonte di verita' per quella catena).
        self.declare_parameter('camera_to_tool0_quat', [0.0418853, 0.0106185, 0.0165691, 0.998929])
        self.declare_parameter('grasp_frame_to_tool0_quat', [0.5, 0.5, 0.5, 0.5])
        self._camera_to_tool0 = Rotation.from_quat(self.get_parameter('camera_to_tool0_quat').value)
        self._grasp_frame_to_tool0 = Rotation.from_quat(self.get_parameter('grasp_frame_to_tool0_quat').value)

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

        # Bbox pixel [x1, y1, x2, y2] dell'oggetto target, dallo stesso frame
        # del target_point corrispondente (pubblicati insieme da
        # sgg_ros_node.py, 22/09/2026). Usata per il crop X/Y lato
        # grasp_server.py -- se non ancora arrivata, il trigger parte comunque
        # senza bbox (solo filtro di profondita', comportamento pre-22/09).
        self.last_target_bbox = None
        self.create_subscription(
            Float32MultiArray, self.get_parameter('target_bbox_topic').get_parameter_value().string_value,
            self.target_bbox_callback, 10
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

    def target_bbox_callback(self, msg: Float32MultiArray):
        self.last_target_bbox = list(msg.data)  # [x1, y1, x2, y2] in pixel

    def _rotation_angle_deg(self, rotation_matrix):
        """Angolo (gradi) di cui dovrebbe ruotare il polso per raggiungere
        questa presa, rispetto alla sua posa ATTUALE -- non serve conoscere
        l'orientamento live del robot per stimarlo: nella composizione usata
        da ET_node (ee_orientation_ * camera_to_tool0_ * q_camera_frame *
        grasp_frame_to_tool0_) l'orientamento attuale moltiplica tutto a
        sinistra, quindi l'angolo tra l'orientamento desiderato e quello
        attuale e' esattamente l'angolo di rotazione del resto della catena,
        che e' quello che calcoliamo qui (stessi ordine e costanti di
        ET_node.cpp, verificati numericamente il 21/09/2026)."""
        q_cam = Rotation.from_matrix(np.array(rotation_matrix))
        delta = self._camera_to_tool0 * q_cam * self._grasp_frame_to_tool0
        return abs(delta.magnitude()) * 180.0 / np.pi

    def _best_equivalent_rotation(self, rotation_matrix):
        """Una presa a due dita simmetriche ha un gemello meccanicamente
        identico, ruotato di 180 gradi attorno al proprio asse di
        avvicinamento (le due dita sono intercambiabili sull'oggetto) --
        GraspNet non elimina questa ridondanza da solo. Restituisce
        (matrice, angolo_gradi) del gemello che richiede la rotazione
        minore rispetto alla posa attuale del polso, cosi' non inseguiamo
        la versione "capovolta" quando quella dritta e' equivalente e
        piu' comoda. Asse X = approccio nel frame di grasp di GraspNet
        (vedi grasp_frame_to_tool0_quat in ET_node), quindi il gemello e'
        una rotazione di 180 gradi attorno a X LOCALE (post-moltiplicazione).
        Aggiunto il 23/09/2026 su segnalazione in laboratorio."""
        q_cam = Rotation.from_matrix(np.array(rotation_matrix))
        q_flip = q_cam * Rotation.from_euler('x', 180, degrees=True)
        angle_orig = self._rotation_angle_deg(q_cam.as_matrix())
        angle_flip = self._rotation_angle_deg(q_flip.as_matrix())
        if angle_flip < angle_orig:
            return q_flip.as_matrix(), angle_flip
        return np.array(rotation_matrix), angle_orig

    def select_grasp(self, result):
        """Tra le prese entro l'apertura della pinza (e, se noto, vicine al
        target), sceglie quella con riorientamento stimato minore tra quelle
        con punteggio sopra min_grasp_score; se nessuna raggiunge la soglia,
        ripiega sulla migliore per punteggio tra tutte quelle valide. None se
        non c'e' nessuna presa valida."""
        grasps = result.get('grasps') or [result]
        # Larghezze grezze di TUTTE le candidate, prima di qualunque filtro
        # (23/09/2026) -- per capire se le prese scartate per larghezza sono
        # concentrate su valori enormi (probabile tavolo dentro il margine del
        # crop) o solo un po' sopra il diametro reale (sovrastima di GraspNet
        # sull'oggetto stesso).
        larghezze_grezze = sorted(g.get('width', float('nan')) for g in grasps)
        self.get_logger().info(
            'Larghezze grezze (m), tutte le candidate: '
            + ', '.join(f'{w:.3f}' for w in larghezze_grezze))
        # Le prese piu' larghe dell'apertura della pinza sono scartate subito:
        # su un oggetto largo (es. bottiglia sdraiata presa sul corpo) GraspNet
        # puo' proporre prese che la Hand-E non riesce a chiudere.
        n_prima = len(grasps)
        grasps = [g for g in grasps if g.get('width', 0.0) <= self.max_grasp_width]
        if len(grasps) < n_prima:
            self.get_logger().info(
                f'Scartate {n_prima - len(grasps)} prese su {n_prima} piu\' larghe di {self.max_grasp_width:.3f} m.')
        if not grasps:
            self.get_logger().warn(
                f'Nessuna presa entro l\'apertura della pinza ({self.max_grasp_width:.3f} m): nessun orientamento pubblicato.')
            return None

        if self.last_target_points:
            near_target = [
                g for g in grasps
                if min(np.linalg.norm(np.array(g['translation']) - p) for p in self.last_target_points) < GRASP_TARGET_RADIUS_M
            ]
            if not near_target:
                best_dist = min(min(np.linalg.norm(np.array(g['translation']) - p) for p in self.last_target_points) for g in grasps)
                self.get_logger().warn(
                    f'Nessuno dei {len(grasps)} grasp entro {GRASP_TARGET_RADIUS_M} m dal target '
                    f'(il piu\' vicino a {best_dist:.3f} m): nessun orientamento pubblicato.')
                return None
            # Log per stadio (22/09/2026): quante sopravvivono a ciascun filtro,
            # non solo il caso zero -- per capire DOVE si restringe la scelta
            # quando il grasp finale non e' quello piu' comodo.
            self.get_logger().info(
                f'{len(near_target)} presa/e su {len(grasps)} entro {GRASP_TARGET_RADIUS_M} m dal target.')
            grasps = near_target
        else:
            self.get_logger().warn('Nessun target noto: scelgo comunque tra tutte le prese valide.')

        quality = [g for g in grasps if g['score'] >= self.min_grasp_score]
        if not quality:
            # Anche nel ripiego (nessuna sopra soglia qualita') va applicato
            # il controllo del gemello a 180 gradi (23/09/2026) -- altrimenti
            # su questo ramo (frequente: capita ogni volta che nessuna presa
            # raggiunge min_grasp_score) si torna a inseguire la versione
            # capovolta anche quando quella dritta sarebbe equivalente.
            fallback = grasps[0]  # gia' ordinate per score dal server
            fb_matrix, fb_angle = self._best_equivalent_rotation(fallback['rotation_matrix'])
            fallback = dict(fallback, rotation_matrix=fb_matrix)
            self.get_logger().warn(
                f"Nessuna presa sopra la soglia di qualita' ({self.min_grasp_score:.2f}): "
                f"scelgo la migliore per punteggio tra le {len(grasps)} valide "
                f"(score={fallback['score']:.3f}, rotazione stimata={fb_angle:.1f} deg).")
            return fallback
        self.get_logger().info(
            f"{len(quality)} presa/e su {len(grasps)} sopra la soglia di qualita' ({self.min_grasp_score:.2f}).")

        # Per ciascuna candidata, la rotazione stimata e' quella del gemello
        # (dritto o capovolto di 180 gradi attorno all'approccio) piu' comodo
        # -- vedi _best_equivalent_rotation().
        equivalents = {id(g): self._best_equivalent_rotation(g['rotation_matrix']) for g in quality}
        ranked = sorted(quality, key=lambda g: equivalents[id(g)][1])
        for g in ranked[:3]:
            self.get_logger().info(
                f"  candidata: score={g['score']:.3f}, larghezza={g.get('width', float('nan')):.3f} m, "
                f"rotazione stimata={equivalents[id(g)][1]:.1f} deg")
        best = ranked[0]
        best_matrix, best_angle = equivalents[id(best)]
        best = dict(best, rotation_matrix=best_matrix)
        self.get_logger().info(
            f"Presa scelta: score={best['score']:.3f}, larghezza={best.get('width', float('nan')):.3f} m, "
            f"rotazione stimata={best_angle:.1f} deg "
            f"(sopra soglia qualita' {self.min_grasp_score:.2f}, tra {len(quality)} candidate).")
        return best

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

        bbox_status = 'bbox nota' if self.last_target_bbox is not None else 'nessuna bbox nota'
        self.get_logger().info(f'Trigger ricevuto ({bbox_status}): chiamo il server GraspNet sulla VM.')

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

        # --- 2b. Bbox del target con margine (22/09/2026) ---
        # Converte il margine in metri (bbox_margin_m, camera frame) in
        # pixel usando gli stessi intrinseci e la Z nota del target (stesso
        # frame camera del bbox, pubblicati insieme da sgg_ros_node.py) --
        # niente a che fare col frame base_link del robot, qui si resta
        # sempre nel frame della camera. Se bbox o target non ancora
        # disponibili, si procede senza (comportamento pre-22/09: solo
        # filtro di profondita' lato server).
        bbox_with_margin = None
        if self.last_target_bbox is not None and self.last_target_points:
            z = float(self.last_target_points[0][2])
            if z > 0:
                margin_px_x = self.bbox_margin_m * fx / z
                margin_px_y = self.bbox_margin_m * fy / z
                x1, y1, x2, y2 = self.last_target_bbox
                bbox_with_margin = [
                    max(0.0, x1 - margin_px_x),
                    max(0.0, y1 - margin_px_y),
                    min(float(depth_img.shape[1]), x2 + margin_px_x),
                    min(float(depth_img.shape[0]), y2 + margin_px_y),
                ]

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
        if bbox_with_margin is not None:
            payload['bbox'] = bbox_with_margin

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
        # Stampa diagnostica dell'orientamento proposto: quaternione grezzo
        # (camera frame, stesso pubblicato su /graspnet/grasp_orientation) +
        # roll/pitch/yaw in gradi, piu' leggibili a occhio in lab.
        rpy_deg = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        self.get_logger().info(
            f"Grasp pubblicato (score={grasp['score']:.3f}): "
            f"quat[xyzw]=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}), "
            f"rpy[deg]=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}) in camera frame."
        )


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