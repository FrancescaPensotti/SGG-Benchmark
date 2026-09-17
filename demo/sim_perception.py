#!/usr/bin/env python3
"""
Percezione e GraspNet simulati, per provare la catena di controllo con il
robot finto (use_fake_hardware:=true) e il Falcon, senza camera ne' VM.

Un oggetto virtuale fisso in base_link viene "visto" da una camera montata su
tool0 con la stessa calibrazione di ur_camera_config.yaml, usando la posa
dell'end-effector di `latency` secondi prima (come i cicli lenti di SGG), con
rumore gaussiano. Pubblica sugli stessi topic dei nodi reali:
  - mode=implicit: /sgg/candidate_targets (PoseArray, un candidato)
  - mode=explicit: /sgg/target_point (PointStamped, serve sgg_to_et_bridge.py)
Al trigger GraspNet risponde dopo `graspnet_delay` secondi con un grasp "dall'alto"
ruotato di `grasp_yaw_deg`, e confronta l'orientamento che ET_node ne ricava
(/debug/desired_ee_orientation) con quello atteso.

Si lancia con il Python di sistema: ros2_setup && python3 demo/sim_perception.py
"""

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PointStamped, QuaternionStamped
from std_msgs.msg import Bool


# Quaternioni in ordine [x, y, z, w], come in ROS.
def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_norm(q):
    return q / np.linalg.norm(q)


def q_rotate(q, v):
    return q_mul(q_mul(q, np.array([v[0], v[1], v[2], 0.0])), q_conj(q))[:3]


def q_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(angle / 2.0)), np.cos(angle / 2.0)])


def q_angle_deg(a, b):
    d = abs(float(np.dot(q_norm(a), q_norm(b))))
    return np.degrees(2.0 * np.arccos(min(1.0, d)))


class SimPerception(Node):
    def __init__(self):
        super().__init__('sim_perception')

        self.declare_parameter('mode', 'implicit')                 # implicit | explicit
        # Oggetto rispetto alla posizione di tool0 al primo messaggio di posa
        # (base_link); ignorato se object_xyz ha 3 valori.
        self.declare_parameter('object_offset_from_start', [0.10, 0.05, -0.40])
        self.declare_parameter('object_xyz', [0.0])
        self.declare_parameter('period', 4.0)                      # ciclo SGG simulato [s]
        self.declare_parameter('latency', 3.0)                     # eta' della posa usata [s]
        self.declare_parameter('noise_std', 0.0)                   # rumore sulla posizione in camera frame [m]
        self.declare_parameter('check_fov', True)
        self.declare_parameter('graspnet_enabled', True)
        self.declare_parameter('graspnet_delay', 2.5)
        self.declare_parameter('grasp_yaw_deg', 30.0)
        # Stessi valori di ur_camera_config.yaml e dei default di ET_node.
        self.declare_parameter('offset_camera_tool0', [-0.0334477, -0.062187, 0.0873308])
        self.declare_parameter('camera_to_tool0_quat', [-0.00163112, 0.00508842, 0.00484543, 0.999974])
        self.declare_parameter('grasp_frame_to_tool0_quat', [0.0, 0.7071068, 0.0, 0.7071068])
        self.declare_parameter('gripper_offset', 0.15)
        self.declare_parameter('hover_clearance', 0.05)

        gp = lambda n: self.get_parameter(n).value
        self.mode = gp('mode')
        self.latency = float(gp('latency'))
        self.noise_std = float(gp('noise_std'))
        self.check_fov = bool(gp('check_fov'))
        self.graspnet_enabled = bool(gp('graspnet_enabled'))
        self.graspnet_delay = float(gp('graspnet_delay'))
        self.offset = np.array(gp('offset_camera_tool0'), dtype=float)
        self.q_cam = q_norm(np.array(gp('camera_to_tool0_quat'), dtype=float))
        self.q_g2t = q_norm(np.array(gp('grasp_frame_to_tool0_quat'), dtype=float))
        self.tool_offset = float(gp('gripper_offset')) + float(gp('hover_clearance'))
        self.object_offset = np.array(gp('object_offset_from_start'), dtype=float)
        obj = gp('object_xyz')
        self.object_xyz = np.array(obj, dtype=float) if len(obj) == 3 else None

        # Grasp atteso in base_link: z di tool0 verso il basso, ruotato attorno alla verticale.
        yaw = np.radians(float(gp('grasp_yaw_deg')))
        self.q_des_expected = q_norm(q_mul(q_axis_angle([0, 0, 1], yaw), q_axis_angle([1, 0, 0], np.pi)))

        self.pose_history = deque(maxlen=20000)
        self.rng = np.random.default_rng()
        self.grasp_timer = None
        self.last_visible_p_cam = None

        self.create_subscription(PoseStamped, '/admittance_controller/pose_debug', self.pose_callback, 50)
        self.create_subscription(Bool, '/graspnet/trigger', self.trigger_callback, 10)
        self.create_subscription(QuaternionStamped, '/debug/desired_ee_orientation', self.desired_orientation_callback, 10)
        self.candidates_pub = self.create_publisher(PoseArray, '/sgg/candidate_targets', 10)
        self.target_pub = self.create_publisher(PointStamped, '/sgg/target_point', 10)
        self.grasp_pub = self.create_publisher(QuaternionStamped, '/graspnet/grasp_orientation', 10)

        self.create_timer(float(gp('period')), self.publish_target)
        self.create_timer(1.0, self.log_hover_distance)

        self.get_logger().info(
            f"Percezione simulata: mode={self.mode}, period={gp('period')}s, latency={self.latency}s, "
            f"noise_std={self.noise_std}m, graspnet={'on' if self.graspnet_enabled else 'off'}")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_callback(self, msg):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = q_norm(np.array([msg.pose.orientation.x, msg.pose.orientation.y,
                             msg.pose.orientation.z, msg.pose.orientation.w]))
        self.pose_history.append((self.now_s(), p, q))
        if self.object_xyz is None:
            self.object_xyz = p + self.object_offset
            self.get_logger().info(f"Oggetto virtuale in base_link: {np.round(self.object_xyz, 3).tolist()}")

    def pose_at(self, t):
        for stamp, p, q in reversed(self.pose_history):
            if stamp <= t:
                return p, q
        return self.pose_history[0][1], self.pose_history[0][2]

    def publish_target(self):
        if not self.pose_history or self.object_xyz is None:
            self.get_logger().warn('Nessuna posa da /admittance_controller/pose_debug: niente target.')
            return

        p_ee, q_ee = self.pose_at(self.now_s() - self.latency)
        p_tool = q_rotate(q_conj(q_ee), self.object_xyz - p_ee)
        p_cam = q_rotate(q_conj(self.q_cam), p_tool - self.offset)
        p_cam = p_cam + self.rng.normal(0.0, self.noise_std, 3) if self.noise_std > 0 else p_cam

        if self.check_fov:
            fx, cx, cy, w, h = 912.1, 650.9, 383.9, 1280, 720
            visible = p_cam[2] > 0.05
            if visible:
                u = fx * p_cam[0] / p_cam[2] + cx
                v = fx * p_cam[1] / p_cam[2] + cy
                visible = 0 <= u < w and 0 <= v < h
            if not visible:
                # Come SGG: l'oggetto resta nel grafo e viene ripubblicato
                # con l'ultima posizione vista.
                if self.last_visible_p_cam is None:
                    self.get_logger().warn('Oggetto mai visto dalla camera simulata: niente target.')
                    return
                self.get_logger().warn('Oggetto fuori dal campo visivo: ripubblico l\'ultima posizione vista (memoria del grafo).')
                p_cam = self.last_visible_p_cam
            else:
                self.last_visible_p_cam = p_cam

        stamp = self.get_clock().now().to_msg()
        if self.mode == 'explicit':
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_color_optical_frame'
            msg.point.x, msg.point.y, msg.point.z = map(float, p_cam)
            self.target_pub.publish(msg)
        else:
            msg = PoseArray()
            msg.header.stamp = stamp
            msg.header.frame_id = 'camera_color_optical_frame'
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, p_cam)
            pose.orientation.w = 1.0
            msg.poses.append(pose)
            self.candidates_pub.publish(msg)
        self.get_logger().info(f'Target pubblicato ({self.mode}), camera frame: {np.round(p_cam, 3).tolist()}')

    def trigger_callback(self, msg):
        if not msg.data or not self.graspnet_enabled:
            return
        self.get_logger().info(f'Trigger GraspNet ricevuto: rispondo tra {self.graspnet_delay}s.')
        if self.grasp_timer is not None:
            self.grasp_timer.cancel()
        self.grasp_timer = self.create_timer(self.graspnet_delay, self.publish_grasp)

    def publish_grasp(self):
        self.grasp_timer.cancel()
        self.grasp_timer = None
        if not self.pose_history:
            return
        _, _, q_ee = self.pose_history[-1]
        # Inverso della composizione di ET_node: q_des = q_ee * q_cam * q_grasp * q_g2t
        q_grasp = q_norm(q_mul(q_mul(q_mul(q_conj(self.q_cam), q_conj(q_ee)), self.q_des_expected), q_conj(self.q_g2t)))
        msg = QuaternionStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w = map(float, q_grasp)
        self.grasp_pub.publish(msg)
        self.get_logger().info('Grasp simulato pubblicato (dall\'alto, yaw impostato).')

    def desired_orientation_callback(self, msg):
        q = np.array([msg.quaternion.x, msg.quaternion.y, msg.quaternion.z, msg.quaternion.w])
        err = q_angle_deg(q, self.q_des_expected)
        esito = 'OK' if err < 5.0 else 'DIVERSO DAL PREVISTO'
        self.get_logger().info(f'Orientamento calcolato da ET_node vs atteso: {err:.1f} gradi -> {esito}')

    def log_hover_distance(self):
        if not self.pose_history or self.object_xyz is None:
            return
        _, p, q = self.pose_history[-1]
        hover = self.object_xyz - q_rotate(q, [0.0, 0.0, self.tool_offset])
        z_axis = q_rotate(q, [0.0, 0.0, 1.0])
        self.get_logger().info(
            f'tool0->hover {np.linalg.norm(p - hover):.3f} m | asse z tool0 {np.round(z_axis, 2).tolist()} | '
            f'errore orientamento vs grasp {q_angle_deg(q, self.q_des_expected):.1f} gradi')


def main():
    rclpy.init()
    node = SimPerception()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
