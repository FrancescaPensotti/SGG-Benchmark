#!/usr/bin/env python3
# demo/moveit_goal_node.py
# ─────────────────────────────────────────────────────────────
# Secondo mattone dell'integrazione MoveIt.
#
# Flusso:
#   1) Riceve la posizione del target dal nodo SGG su un topic
#      (/sgg/target_point), espressa nel frame della camera.
#   2) La trasforma in base_link con TF2 (richiede la static
#      transform tool0 -> camera_color_optical_frame attiva).
#   3) Aggiunge un offset di sicurezza in Z (si ferma SOPRA
#      l'oggetto, non ci va sopra).
#   4) Invia un goal di posizione a MoveIt (move_group).
#
# SICUREZZA:
#   - Parte in modalita' plan_only = True: MoveIt PIANIFICA e
#     mostra la traiettoria in RViz ma NON muove il robot.
#     Quando si è sicuri, si lancia con plan_only:=false per eseguire.
#   - Velocita' e accelerazione ridotte al 10%.
#   - Offset di sicurezza di 10 cm sopra il target.
#
# USO in lab (con robot + camera + static transform + move_group):
#   # solo pianificazione (sicuro, default):
#   python3 demo/moveit_goal_node.py
#   # esecuzione reale:
#   ros2 run ... --ros-args -p plan_only:=false
#   (oppure cambiare il default del parametro qui sotto)
# ─────────────────────────────────────────────────────────────

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import tf2_ros
from tf2_geometry_msgs import do_transform_point  # registra transform per PointStamped
from geometry_msgs.msg import PointStamped

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, PositionConstraint,
    BoundingVolume, PlanningOptions,
)
from shape_msgs.msg import SolidPrimitive

# ── Configurazione ──────────────────────────────────────────
PLANNING_GROUP = "manipulator"               # gruppo del braccio UR5e
BASE_FRAME     = "base_link"                  # frame goal per MoveIt
EE_LINK        = "tool0"                      # link che vogliamo portare sul target
TARGET_TOPIC   = "/sgg/target_point"          # da dove arriva la posizione (nodo SGG)
ACTION_NAME    = "/move_action"               # action server di move_group


class MoveItGoalNode(Node):
    def __init__(self):
        super().__init__('moveit_goal_node')

        # Parametri (modificabili da riga di comando con -p nome:=valore)
        self.declare_parameter('plan_only', True)          # True = non muove, solo pianifica
        self.declare_parameter('safety_z_offset', 0.10)    # metri sopra l'oggetto
        self.declare_parameter('vel_scaling', 0.1)         # 10% velocita'
        self.declare_parameter('acc_scaling', 0.1)         # 10% accelerazione

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Action client verso move_group
        self.client = ActionClient(self, MoveGroup, ACTION_NAME)

        # Subscriber: riceve la posizione del target dal nodo SGG
        self.sub = self.create_subscription(
            PointStamped, TARGET_TOPIC, self.on_target, 10)

        self.busy = False  # evita di sovrapporre piu' goal

        plan_only = self.get_parameter('plan_only').value
        modo = "SOLO PIANIFICAZIONE (non muove)" if plan_only else "ESECUZIONE REALE"
        self.get_logger().info(f"MoveIt Goal Node avviato — modalita': {modo}")
        self.get_logger().info(f"In ascolto su '{TARGET_TOPIC}'")

    def on_target(self, msg: PointStamped):
        if self.busy:
            self.get_logger().warn("Goal precedente ancora in corso, ignoro.")
            return

        # ── 1. Trasforma il punto dal frame camera a base_link ──
        try:
            punto_base = self.tf_buffer.transform(
                msg, BASE_FRAME,
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f"Trasformazione TF fallita: {e}")
            return

        # ── 2. Offset di sicurezza in Z (si ferma sopra l'oggetto) ──
        z_off = self.get_parameter('safety_z_offset').value
        x = punto_base.point.x
        y = punto_base.point.y
        z = punto_base.point.z + z_off

        self.get_logger().info(
            f"Target in base_link: ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"[con offset Z +{z_off} m]")

        # ── 3. Costruisci e invia il goal a MoveIt ──
        self.invia_goal(x, y, z)

    def invia_goal(self, x, y, z):
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                f"move_group non disponibile su '{ACTION_NAME}'. "
                "E' attivo il robot/MoveIt?")
            return

        # --- Vincolo di posizione: una piccola sfera centrata sul target ---
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.01]  # raggio 1 cm di tolleranza

        region = BoundingVolume()
        region.primitives.append(primitive)
        from geometry_msgs.msg import Pose
        sphere_pose = Pose()
        sphere_pose.position.x = x
        sphere_pose.position.y = y
        sphere_pose.position.z = z
        sphere_pose.orientation.w = 1.0
        region.primitive_poses.append(sphere_pose)

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = BASE_FRAME
        pos_constraint.link_name = EE_LINK
        pos_constraint.constraint_region = region
        pos_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pos_constraint)

        # --- Vincolo di orientamento: end-effector rivolto verso il basso ---
        # tool0 con asse Z verso il tavolo (approccio dall'alto).
        # Quaternione (-0.707, 0.707, 0, 0) = orientamento reale di tool0 letto da
        # tf2_echo in lab (RPY [180°, 0°, -90°]), non (1,0,0,0) teorico, per l'offset
        # di montaggio del tool. Validato: inizio/fine traiettoria corretti; il "giro"
        # intermedio è dovuto al planning OMPL nello spazio dei giunti (non cartesiano) —
        # verrà risolto passando a pianificazione cartesiana ad attrattori (vedi Energy-Tanks).
        from moveit_msgs.msg import OrientationConstraint
        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = BASE_FRAME
        ori_constraint.link_name = EE_LINK
        ori_constraint.orientation.x = -0.707
        ori_constraint.orientation.y = 0.707
        ori_constraint.orientation.z = 0.0
        ori_constraint.orientation.w = 0.0
        ori_constraint.absolute_x_axis_tolerance = 0.3
        ori_constraint.absolute_y_axis_tolerance = 0.3
        ori_constraint.absolute_z_axis_tolerance = 0.3
        ori_constraint.weight = 1.0
        constraints.orientation_constraints.append(ori_constraint)

        # --- Richiesta di pianificazione ---
        req = MotionPlanRequest()
        req.group_name = PLANNING_GROUP
        req.goal_constraints.append(constraints)
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = self.get_parameter('vel_scaling').value
        req.max_acceleration_scaling_factor = self.get_parameter('acc_scaling').value

        # --- Opzioni: plan_only decide se eseguire o solo pianificare ---
        options = PlanningOptions()
        options.plan_only = self.get_parameter('plan_only').value

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = options

        self.busy = True
        self.get_logger().info("Invio goal a MoveIt...")
        future = self.client.send_goal_async(goal)
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rifiutato da MoveIt.")
            self.busy = False
            return
        self.get_logger().info("Goal accettato, attendo il risultato...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future):
        result = future.result().result
        code = result.error_code.val
        if code == 1:  # SUCCESS
            self.get_logger().info("✓ Pianificazione/esecuzione riuscita.")
        else:
            self.get_logger().warn(f"MoveIt ha restituito error_code = {code}")
        self.busy = False


def main():
    rclpy.init()
    node = MoveItGoalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()