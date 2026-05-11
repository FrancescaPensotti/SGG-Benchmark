import torch
import clip
import cv2
import numpy as np
from PIL import Image
import sys
import os

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.onnx_model import SGG_ONNX_Model
from demo.query_scene import get_position_history, get_current_position, get_path_to_targets_meters

# ── Configurazione ──────────────────────────────────────────
ONNX_PATH = "checkpoints/VG150/react++_yolov8m/model.onnx"
SIMILARITY_THRESHOLD = 0.85
FREQ_THRESHOLD = 10
DISAPPEAR_THRESHOLD = 30
SCALA_PIXEL_METRI = 0.00105  # da calibrare in lab

BLACKLIST_OBJECTS = {
    'table', 'floor', 'wall', 'ceiling', 'chair',
    'ground', 'background', 'window', 'door',
    'hair', 'nose', 'face', 'head', 'eye', 'ear',
    'mouth', 'neck', 'arm', 'leg'
}

# ── Carica i modelli ────────────────────────────────────────
print("Carico SGG-Benchmark...")
sgg = SGG_ONNX_Model(None, ONNX_PATH)

print("Carico CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"Pronti! Uso: {device}")

# ── Albero semantico ────────────────────────────────────────
scene_graph = []

# ── Funzioni (stesse di incremental_scene.py) ───────────────
def get_clip_embedding(image, box):
    x1, y1, x2, y2 = map(int, box[:4])
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(crop_tensor)
        emb = emb / emb.norm()
    return emb

def find_existing_node(label, embedding):
    for i, node in enumerate(scene_graph):
        if node['label'] == label:
            sim = torch.cosine_similarity(node['embedding'], embedding).item()
            if sim >= SIMILARITY_THRESHOLD:
                return i
    return -1

def update_scene_graph(image, bboxes, rels):
    box_to_node = {}

    for i, box in enumerate(bboxes):
        cls_idx = int(box[5])
        label = sgg.stats['obj_classes'].get(cls_idx, str(cls_idx))

        if label in BLACKLIST_OBJECTS:
            continue

        embedding = get_clip_embedding(image, box)
        if embedding is None:
            continue

        existing_idx = find_existing_node(label, embedding)

        x1, y1, x2, y2 = map(int, box[:4])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        if existing_idx >= 0:
            scene_graph[existing_idx]['count'] += 1
            scene_graph[existing_idx]['embedding'] = embedding
            scene_graph[existing_idx]['position'] = (cx, cy)
            scene_graph[existing_idx]['position_history'].append((cx, cy))
            scene_graph[existing_idx]['frames_not_seen'] = 0
            box_to_node[i] = existing_idx
        else:
            new_node = {
                'label': label,
                'embedding': embedding,
                'relazioni': {},
                'count': 1,
                'position': (cx, cy),
                'position_history': [(cx, cy)],
                'frames_not_seen': 0
            }
            scene_graph.append(new_node)
            box_to_node[i] = len(scene_graph) - 1

    if rels is not None and len(rels) > 0:
        for rel in rels:
            subj_box = int(rel[0])
            obj_box = int(rel[1])
            rel_idx = int(rel[2])
            pred = sgg.stats['rel_classes'].get(rel_idx, str(rel_idx))

            if subj_box in box_to_node and obj_box in box_to_node:
                subj_node = box_to_node[subj_box]
                obj_node = box_to_node[obj_box]

                rel_key = (pred, obj_node)
                if rel_key in scene_graph[subj_node]['relazioni']:
                    scene_graph[subj_node]['relazioni'][rel_key] += 1
                else:
                    scene_graph[subj_node]['relazioni'][rel_key] = 1

    seen_nodes = set(box_to_node.values())
    to_remove = []
    for i, node in enumerate(scene_graph):
        if i not in seen_nodes:
            node['frames_not_seen'] += 1
            if node['frames_not_seen'] >= DISAPPEAR_THRESHOLD:
                print(f"  ⚠️ '{node['label']}' sparito dalla scena — probabilmente preso dal robot.")
                to_remove.append(i)

    for i in sorted(to_remove, reverse=True):
        scene_graph.pop(i)

def print_scene_graph():
    print("\n" + "="*50)
    print(f"ALBERO SEMANTICO — {len(scene_graph)} oggetti nella scena")
    print("="*50)
    for i, node in enumerate(scene_graph):
        if node['count'] < FREQ_THRESHOLD:
            continue
        pos = node.get('position', 'N/A')
        history = node.get('position_history', [])
        print(f"  [{i}] {node['label']} (visto {node['count']} volte) — pos attuale: {pos} — storia: {len(history)} punti")
        for (pred, obj_idx), count in node['relazioni'].items():
            if count < FREQ_THRESHOLD:
                continue
            if pred not in VG_TO_FUNCTIONAL and pred not in VG_TO_SPATIAL:
                continue
            print(f"       --({pred})--> {scene_graph[obj_idx]['label']} [vista {count}x]")
    print("="*50 + "\n")

# ── Nodo ROS2 ───────────────────────────────────────────────
class SGGNode(Node):
    def __init__(self):
        super().__init__('sgg_node')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.img = None

        # Subscriber al topic della RealSense
        self.subscription = self.create_subscription(
            RosImage,
            '/camera/color/image_raw',
            self.frame_callback,
            10)
        
        self.get_logger().info("SGG Node avviato — in ascolto su /camera/color/image_raw")

    def frame_callback(self, msg):
        """Callback chiamata ad ogni nuovo frame dalla camera."""
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_count += 1

        # Elabora ogni 5 frame
        if self.frame_count % 5 == 0:
            result = sgg.predict(frame, visu_type='video')
            self.img, dbg = result

            if dbg is not None:
                bboxes, rels = dbg
                update_scene_graph(frame, bboxes, rels)

        # Mostra il frame
        if self.img is not None:
            cv2.imshow("SGG ROS2 Node", self.img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print_scene_graph()
        elif key == ord('t'):
            label_input = input("Oggetti target (separati da virgola): ")
            target_labels = [l.strip() for l in label_input.split(',')]
            path = get_path_to_targets_meters(scene_graph, target_labels, SCALA_PIXEL_METRI)
            if path:
                print("\nPATH VERSO GLI OBIETTIVI:")
                for step, target in enumerate(path):
                    print(f"  Step {step+1}: {target['label']} → pixel: {target['position_pixel']} | metri: {target['position_metri']}")
            else:
                print("Nessun oggetto trovato.")
        elif key == ord('h'):
            label = input("Di quale oggetto vuoi la history? ")
            history = get_position_history(scene_graph, label)
            if history:
                print(f"History di '{label}': {history}")
            else:
                print(f"Oggetto '{label}' non trovato.")
        elif key == ord('q'):
            self.get_logger().info("Chiusura nodo SGG.")
            rclpy.shutdown()

# ── Main ────────────────────────────────────────────────────
def main():
    rclpy.init()
    node = SGGNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()