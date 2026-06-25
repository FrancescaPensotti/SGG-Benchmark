#!/usr/bin/env python3
# demo/test_tf_transform.py
# ─────────────────────────────────────────────────────────────
# Primo mattone per l'integrazione MoveIt.
#
# Scopo: prendere un punto 3D nel frame della camera
# (camera_color_optical_frame) e trasformarlo nel frame base del
# robot (base_link) usando TF2.
#
# Questo NON muove il robot. Serve solo a verificare che:
#   1) l'albero TF sia connesso
#   2) la trasformazione dia coordinate sensate
#
# Una volta che questo funziona, sappiamo che possiamo portare
# le posizioni degli oggetti dal frame camera al frame base,
# che e' il prerequisito per dare un goal a MoveIt.
#
# USO in lab (con robot + camera + static_transform_publisher attivi):
#   python3 demo/test_tf_transform.py
# ─────────────────────────────────────────────────────────────

import rclpy
from rclpy.node import Node

import tf2_ros
from geometry_msgs.msg import PointStamped
# import necessario: registra il "do_transform" per i PointStamped
from tf2_geometry_msgs import do_transform_point

# ── Frame coinvolti ─────────────────────────────────────────
SOURCE_FRAME = "camera_color_optical_frame"   # dove vive il punto (camera)
TARGET_FRAME = "base_link"                     # dove lo vogliamo (robot)

# ── Punto di test nel frame camera (x, y, z in metri) ───────
# Per ora un valore finto, plausibile per un oggetto visto dalla
# camera: ~9 cm a destra, ~5 cm in basso, ~46 cm di profondita'.
# In seguito qui arrivera' la posizione vera dal nodo SGG.
PUNTO_TEST = (0.09, 0.05, 0.46)


class TFTester(Node):
    def __init__(self):
        super().__init__('tf_tester')

        # Buffer e listener TF: raccolgono le trasformazioni pubblicate
        # nel sistema (robot, camera, static transform).
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Prova a trasformare il punto ogni secondo.
        self.timer = self.create_timer(1.0, self.prova_trasformazione)

        self.get_logger().info(
            f"TF Tester avviato. Trasformo da '{SOURCE_FRAME}' a '{TARGET_FRAME}'."
        )

    def prova_trasformazione(self):
        # Costruisci il punto nel frame camera.
        punto_camera = PointStamped()
        punto_camera.header.frame_id = SOURCE_FRAME
        punto_camera.header.stamp = rclpy.time.Time().to_msg()  # tempo 0 = "ultima TF disponibile"
        punto_camera.point.x = PUNTO_TEST[0]
        punto_camera.point.y = PUNTO_TEST[1]
        punto_camera.point.z = PUNTO_TEST[2]

        try:
            # Chiede a TF2 la trasformazione e la applica al punto.
            punto_base = self.tf_buffer.transform(
                punto_camera,
                TARGET_FRAME,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            p = punto_base.point
            print("\n" + "="*50)
            print("TRASFORMAZIONE RIUSCITA")
            print("="*50)
            print(f"  Punto nel frame camera ({SOURCE_FRAME}):")
            print(f"    x={PUNTO_TEST[0]:.4f}  y={PUNTO_TEST[1]:.4f}  z={PUNTO_TEST[2]:.4f}")
            print(f"  Punto nel frame base ({TARGET_FRAME}):")
            print(f"    x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}")
            print("="*50)

        except tf2_ros.LookupException as e:
            self.get_logger().warn(
                f"Frame non trovato (albero TF non connesso?): {e}"
            )
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"Problema temporale TF: {e}")
        except Exception as e:
            self.get_logger().warn(f"Trasformazione fallita: {e}")


def main():
    rclpy.init()
    node = TFTester()
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