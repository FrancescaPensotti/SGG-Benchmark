import torch
import clip
import cv2
import numpy as np
from PIL import Image
import sys
import os
import threading
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.onnx_model import SGG_ONNX_Model
from demo.query_scene import get_position_history, get_current_position, get_path_to_targets_meters
from demo.aruco_detector import detect_aruco, pixel_to_meters_3d, CAMERA_MATRIX, calcola_scala_da_aruco

# ── Configurazione ──────────────────────────────────────────
ONNX_PATH = "checkpoints/VG150/react++_yolov8m/model.onnx"
SIMILARITY_THRESHOLD = 0.85
FREQ_THRESHOLD = 2
# Fattore di decay: ad ogni ciclo in cui il nodo non viene rivisto, la sua confidenza
# viene moltiplicata per questo valore (vicino a 1).
# Con 0.98: dopo 100 cicli non visti confidence ≈ 0.13, dopo 150 cicli ≈ 0.05.
CONFIDENCE_DECAY = 0.98

# Soglia sotto la quale il nodo viene rimosso dal grafo (oggetto considerato "perso" davvero).
CONFIDENCE_REMOVE_THRESHOLD = 0.05
SCALA_PIXEL_METRI = 0.000435  # fallback se ArUco non visibile, calcolato con z=0.397m

# Margine dal bordo dell'immagine (in pixel) entro cui un oggetto è considerato
# "vicino al bordo" — se l'ultima posizione nota di un nodo era in questa fascia,
# è più plausibile che sia appena uscito dal campo visivo (camera eye-in-hand che
# si è spostata) piuttosto che effettivamente sparito dalla scena.
EDGE_MARGIN_PX = 40

BLACKLIST_OBJECTS = {
    'floor', 'wall', 'ceiling', 'chair',
    'ground', 'background', 'window', 'door',
    'hair', 'nose', 'table','face', 'head', 'eye', 'ear',
    'mouth', 'neck', 'arm', 'leg'
}

# ── Mappe relazioni VG150 → MomaGraph ───────────────────────
VG_TO_FUNCTIONAL = {
    'holding':      'control',
    'using':        'control',
    'carrying':     'control',
    'attached to':  'pairwith',
    'connected to': 'pairwith',
    'hanging from': 'pairwith',
    'plugged into': 'providepower',
    'covering':     'openorclose',
    'on':           'pairwith',
    'in':           'pairwith',
}

VG_TO_SPATIAL = {
    'above':       'higher_than',
    'below':       'lower_than',
    'behind':      'behind',
    'in front of': 'in_front_of',
    'left of':     'left_of',
    'right of':    'right_of',
    'near':        'close',
    'next to':     'close',
    'on':          'touching',
    'in':          'touching',
    'touching':    'touching',
}

# ── Carica i modelli ────────────────────────────────────────
print("Carico SGG-Benchmark...")
sgg = SGG_ONNX_Model(None, ONNX_PATH)

print("Carico CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"Pronti! Uso: {device}")

# ── Albero semantico ────────────────────────────────────────
# TODO (audit): scene_graph e' mutato dentro frame_callback SOTTO self.lock
# (vedi SGGNode.frame_callback), ma letto/mutato senza lock da command_loop
# (gli handler p/h/t/g/v leggono scene_graph direttamente) e da
# gripper_status_callback -> advance_to_next_object. command_loop gira su un
# threading.Thread separato dal contesto rclpy: un decay/pop concorrente in
# frame_callback mentre un handler sta iterando scene_graph puo' leggere stato
# inconsistente o sollevare un'eccezione. Da proteggere con lo stesso self.lock
# anche in command_loop e in gripper_status_callback/advance_to_next_object.
scene_graph = []
current_z = None  # aggiornata quando viene rilevato un marker ArUco

# ── Funzioni ────────────────────────────────────────────────
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
    frame_height, frame_width = image.shape[:2]

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
            scene_graph[existing_idx]['bbox'] = (x1, y1, x2, y2)
            scene_graph[existing_idx]['position_history'].append((cx, cy))
            scene_graph[existing_idx]['position_history'] = scene_graph[existing_idx]['position_history'][-50:]
            scene_graph[existing_idx]['frames_not_seen'] = 0
            scene_graph[existing_idx]['confidence'] = 1.0          # <-- piena confidenza quando rivisto
            box_to_node[i] = existing_idx
        else:
            new_node = {
                'label': label,
                'embedding': embedding,
                'relazioni': {},
                'count': 1,
                'position': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'position_history': [(cx, cy)],
                'frames_not_seen': 0,
                'confidence': 1.0   
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


    # TODO (estensione futura): il decay qui sotto è "cieco" — si applica sempre,
    # senza distinguere PERCHÉ il nodo non è stato visto in questo ciclo. Andrebbe
    # invece congelato (non applicato) quando l'oggetto è plausibilmente ancora
    # presente ma semplicemente non visibile ora, cioè:
    #   (a) fuori dal field of view della camera (la camera si è spostata, eye-in-hand),
    #   (b) occluso da un ostacolo o dal gripper durante il grasping.
    # Per (a): nota il fatto che conosciamo la posa nota della camera via TF
    # (base_link -> camera_color_optical_frame); si potrebbe proiettare la
    # posizione 3D nota del nodo nel frame camera corrente e controllare se cade
    # dentro i limiti dell'immagine — se fuori, congelare il decay per questo nodo.
    # Per (b): servirebbe sapere se il gripper sta transitando sopra la posizione
    # nota del nodo (serve la posa del gripper, non solo la camera).

    seen_nodes = set(box_to_node.values())
    to_remove = []
    for i, node in enumerate(scene_graph):
        if i not in seen_nodes:
            node['frames_not_seen'] += 1

            # Euristica semplificata per il campo visivo (vedi TODO sotto per la
            # versione corretta, con vera geometria 3D). Qui usiamo solo la
            # posizione in pixel dell'ultima volta che il nodo è stato visto:
            # se era vicino al bordo dell'immagine, congeliamo il decay per
            # questo ciclo invece di applicarlo — l'oggetto potrebbe essere
            # semplicemente appena uscito dall'inquadratura, non sparito.
            pos = node.get('position')
            near_edge = False
            if pos is not None:
                px, py = pos
                near_edge = (
                    px < EDGE_MARGIN_PX or px > (frame_width - EDGE_MARGIN_PX) or
                    py < EDGE_MARGIN_PX or py > (frame_height - EDGE_MARGIN_PX)
)
            if not near_edge:
                node['confidence'] *= CONFIDENCE_DECAY

            if node['confidence'] < CONFIDENCE_REMOVE_THRESHOLD:
                print(f"  ⚠️ '{node['label']}' sparito dalla scena — probabilmente preso dal robot.")
                to_remove.append(i)

    # TODO (estensione futura, versione corretta): questa euristica sul bordo
    # dell'immagine è un'approssimazione — un oggetto occluso al centro
    # dell'inquadratura (es. dal gripper durante il grasping) non viene
    # riconosciuto e continua comunque a decadere. La versione corretta
    # richiederebbe: (1) salvare anche la posizione 3D in base_link di ogni nodo
    # (oggi il grafo conosce solo pixel/posizione relativa alla camera), il che
    # richiede una sottoscrizione TF in questo nodo Python (non presente oggi);
    # (2) ad ogni ciclo, per i nodi non visti, riproiettare quella posizione nota
    # nel frame camera CORRENTE (TF aggiornata) e verificare se cade dentro i
    # limiti dell'immagine — a quel punto si distinguerebbe correttamente anche
    # il caso di occlusione (oggetto dietro al gripper), non solo il bordo.

    for i in sorted(to_remove, reverse=True):
        scene_graph.pop(i)

def print_scene_graph():
    print("\n" + "="*50)
    print(f"ALBERO SEMANTICO — {len(scene_graph)} oggetti nella scena")
    print("="*50)
    for i, node in enumerate(scene_graph):
        if node['count'] < FREQ_THRESHOLD:
            continue
        pos = node.get('position', (0, 0))
        history = node.get('position_history', [])
        bbox = node.get('bbox', None)
        if current_z is not None:
            pos_metri = pixel_to_meters_3d(pos[0], pos[1], current_z, CAMERA_MATRIX)
            depth_info = f"z={current_z:.3f}m (ArUco)"
        else:
            pos_metri = (pos[0] * SCALA_PIXEL_METRI, pos[1] * SCALA_PIXEL_METRI)
            depth_info = "z=N/A (scala fissa)"
        print(f"  [{i}] {node['label']} (visto {node['count']} volte) — storia: {len(history)} punti")
        print(f"       pos pixel: {pos} | pos metri: ({pos_metri[0]:.3f}m, {pos_metri[1]:.3f}m) | {depth_info}")
        if bbox:
            print(f"       bbox: {bbox} — larghezza: {bbox[2]-bbox[0]}px, altezza: {bbox[3]-bbox[1]}px")
        for (pred, obj_idx), count in node['relazioni'].items():
            if count < FREQ_THRESHOLD:
                continue
            if pred not in VG_TO_FUNCTIONAL and pred not in VG_TO_SPATIAL:
                continue
            print(f"       --({pred})--> {scene_graph[obj_idx]['label']} [vista {count}x]")
    print("="*50 + "\n")

def get_candidate_targets():
    """Restituisce la lista di candidati da inviare a ET_node per la
    combinazione pesata multi-target: tutti gli oggetti nel grafo con
    confidenza sufficiente (cold start / nessun grasp recente in corso). Il
    filtro di raggio dalla posizione del braccio lo applica ET_node, che la
    conosce (qui non c'è la posa del braccio, solo il grafo di scena).

    NOTA: in precedenza questa funzione accettava anche un parametro
    grasped_label per restituire solo gli oggetti con relazione funzionale
    verso il nodo appena graspato (caso "post-grasp"), ma quel ramo non era
    mai stato chiamato con un valore diverso da None in questo file: la
    stessa identica logica ("cerca la relazione funzionale con conteggio più
    alto verso il nodo graspato") è implementata — ed effettivamente usata —
    in advance_to_next_object() qui sotto. Rimosso per non avere due
    implementazioni indipendenti della stessa cosa, di cui una morta."""

    return [
        {'label': n['label'], 'position': n['position']}
        for n in scene_graph
        if n['count'] >= FREQ_THRESHOLD and n.get('confidence', 0.0) >= CONFIDENCE_REMOVE_THRESHOLD
    ]

# ── Nodo ROS2 ───────────────────────────────────────────────
class SGGNode(Node):
    def __init__(self):
        super().__init__('sgg_node')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.img = None
        self.lock = threading.Lock()
        self.active_target_label = None   # <--label del target da ripubblicare ad ogni frame
        # TODO (audit): '/sgg/candidate_targets' e' un letterale qui, che deve
        # combaciare col default del parametro candidate_targets_topic dichiarato
        # in ET_node.cpp (repo Energy-Tanks) — stessa cosa per '/sgg/target_point'
        # più sotto rispetto a sgg_to_et_bridge.py (parametro input_topic) e
        # moveit_goal_node.py (TARGET_TOPIC). Nessun file di configurazione
        # condiviso tra le due repo: un refuso in uno dei due punti rompe il
        # collegamento senza errori a runtime (il subscriber semplicemente non
        # riceve mai nulla).
        self.candidates_pub = self.create_publisher(PoseArray, '/sgg/candidate_targets', 10)

        
        self.gripper_sub = self.create_subscription(
    Bool, '/gripper/grasp_confirmed', self.gripper_status_callback, 10)

        # Subscriber RealSense
        self.subscription = self.create_subscription(
            RosImage,
            '/camera/camera/color/image_raw',
            self.frame_callback,
            10)
        
        # Publisher della posizione del target verso il nodo MoveIt
        self.target_pub = self.create_publisher(PointStamped, '/sgg/target_point', 10)

        # Timer per visualizzazione (ogni 100ms)
        self.create_timer(0.1, self.display_callback)

        # Thread separato per i comandi da tastiera
        self.cmd_thread = threading.Thread(target=self.command_loop, daemon=True)
        self.cmd_thread.start()

        self.get_logger().info("SGG Node avviato — in ascolto su /camera/camera/color/image_raw")

    def frame_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.frame_count += 1

        if self.frame_count % 5 == 0:
            result = sgg.predict(frame, visu_type='video')
            with self.lock:
                self.img, dbg = result
                if dbg is not None:
                    bboxes, rels = dbg
                    update_scene_graph(frame, bboxes, rels)

                # Rileva ArUco e aggiorna Z e scala
                aruco_results, _, _ = detect_aruco(frame)
                if aruco_results:
                    global current_z, SCALA_PIXEL_METRI
                    current_z = aruco_results[0]['z']
                    SCALA_PIXEL_METRI = calcola_scala_da_aruco(aruco_results[0]['corners'])
                    # Calibrazione: distanza tra due marker
                    if len(aruco_results) >= 2:
                        p1 = aruco_results[0]['position']
                        p2 = aruco_results[1]['position']
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                        print(f"  📏 Distanza tra marker: {dist:.3f}m (attesa: 0.20m)")

                self.republish_active_target()   # ripubblica il target attivo, se c'è
                self.republish_candidate_targets()   # candidati multipli, solo se nessun comando esplicito attivo

    def display_callback(self):
        with self.lock:
            if self.img is not None:
                cv2.imshow("SGG ROS2 Node", self.img)
                cv2.waitKey(1)
    
    def pubblica_target(self, pos_pixel):
        """Pubblica la posizione 3D del target nel frame camera, per il nodo MoveIt."""
        if current_z is None:
            self.get_logger().warn("current_z non disponibile (ArUco non visto): non pubblico il target.")
            return
        pm = pixel_to_meters_3d(pos_pixel[0], pos_pixel[1], current_z, CAMERA_MATRIX)
        msg = PointStamped()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(pm[0])
        msg.point.y = float(pm[1])
        msg.point.z = float(current_z)
        self.target_pub.publish(msg)
        print(f"  → Target pubblicato su /sgg/target_point: ({pm[0]:.3f}, {pm[1]:.3f}, {current_z:.3f}) [frame camera]")

    def gripper_status_callback(self, msg: Bool):
        """Ogni messaggio su questo topic è già un grasp confermato (ET_node ha
        già verificato transizione + distanza) — nessuna logica di transizione
        necessaria qui."""
        if self.active_target_label is None:
            return
        self.advance_to_next_object(self.active_target_label)

    def advance_to_next_object(self, grasped_label):
        """Cerca tra le relazioni del nodo appena graspato quella funzionale
        con conteggio più alto, e la imposta come nuovo target attivo."""
        grasped_node = next((n for n in scene_graph if n['label'] == grasped_label), None)

        if grasped_node is None:
            print(f"  ⚠️ '{grasped_label}' non più nel grafo, nessun proseguimento automatico.")
            self.active_target_label = None
            return

        best_rel, best_count = None, 0
        for (pred, obj_idx), count in grasped_node['relazioni'].items():
            if pred in VG_TO_FUNCTIONAL and count > best_count:
                best_rel, best_count = (pred, obj_idx), count

        if best_rel is None:
            print(f"  ℹ️ Nessuna relazione funzionale per '{grasped_label}': nessun successivo.")
            self.active_target_label = None
            return

        pred, obj_idx = best_rel
        self.active_target_label = scene_graph[obj_idx]['label']
        print(f"  → Grasp di '{grasped_label}' rilevato. Prossimo target: '{self.active_target_label}' ({pred}, {best_count}x)")


    def republish_active_target(self):
        """Ripubblica automaticamente la posizione del target attivo (selezionato con
        l'ultimo comando 't' o 'g'), finché resta nella scena con confidenza sufficiente.
        Se il target esce dalla scena o scende sotto soglia, semplicemente non pubblica
        nulla — lato ET_node questo fa scadere aruco_active in modo naturale, collegando
        la logica di planning al forgetting factor lato percezione."""
        if self.active_target_label is None:
            return

        for node in scene_graph:
            if node['label'] == self.active_target_label and node['count'] >= FREQ_THRESHOLD:
                if node.get('confidence', 0.0) >= CONFIDENCE_REMOVE_THRESHOLD:
                    pos = node.get('position')
                    if pos:
                        self.pubblica_target(pos)
                return
        # Target non trovato: rimosso per decay, oppure mai stato visto — non pubblichiamo.
        

    def republish_candidate_targets(self):
        """Pubblica i candidati multipli SOLO quando non c'è un comando esplicito
        attivo (active_target_label is None) — altrimenti il comando t/g ha
        priorità e usa il canale a target singolo esistente (republish_active_target).
        Non richiede current_z: le posizioni pixel vengono convertite in camera
        frame usando la stessa logica di pubblica_target, per ogni candidato."""
        if self.active_target_label is not None:
            return
        if current_z is None:
            return

        candidates = get_candidate_targets()
        if not candidates:
            return

        msg = PoseArray()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()

        for cand in candidates:
            pos_pixel = cand['position']
            pm = pixel_to_meters_3d(pos_pixel[0], pos_pixel[1], current_z, CAMERA_MATRIX)
            pose = Pose()
            pose.position.x = float(pm[0])
            pose.position.y = float(pm[1])
            pose.position.z = float(current_z)
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.candidates_pub.publish(msg)



        
    def command_loop(self):
        print("\nComandi disponibili:")
        print("  p → stampa albero semantico")
        print("  h → history posizioni di un oggetto")
        print("  t → path verso oggetti target")
        print("  g → target in linguaggio naturale (Gemini)")
        print("  v → verifica posizioni (distanze a coppie + z)")
        print("  q → esci\n")

        while rclpy.ok():
            try:
                cmd = input("Comando: ").strip().lower()
                if cmd == 'p':
                    print_scene_graph()
                elif cmd == 'h':
                    descrizione = input("Di quale oggetto vuoi la history? (puoi descriverlo a parole tue) ")
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    scene_json = scene_graph_to_json(scene_graph, FREQ_THRESHOLD)
                    labels = resolve_targets(descrizione, scene_json)
                    label = labels[0] if labels else descrizione
                    history = get_position_history(scene_graph, label)
                    if history:
                        print(f"History di '{label}': {history}")
                    else:
                        print(f"Oggetto '{label}' non trovato nella scena.")
                elif cmd == 't':
                    descrizione = input("Oggetti target (puoi descriverli a parole tue, separati da virgola): ")
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    scene_json = scene_graph_to_json(scene_graph, FREQ_THRESHOLD)
                    target_labels = resolve_targets(descrizione, scene_json)
                    if not target_labels:
                        # fallback: tratta l'input come label esatte
                        target_labels = [l.strip() for l in descrizione.split(',')]
                    # Il planning gestisce un solo target alla volta: il primo diventa "attivo" e
                    # viene ripubblicato automaticamente ad ogni frame (vedi republish_active_target).
                    # Gli altri restano solo nel path stampato, in attesa della gestione multi-goal.
                    self.active_target_label = target_labels[0] if target_labels else None

                    if current_z is not None:
                        path = []
                        for label in target_labels:
                            for node in scene_graph:
                                if node['label'] == label and node['count'] >= FREQ_THRESHOLD:
                                    pos = node.get('position')
                                    if pos:
                                        pos_metri = pixel_to_meters_3d(pos[0], pos[1], current_z, CAMERA_MATRIX)
                                        path.append({'label': label, 'position_pixel': pos, 'position_metri': pos_metri})
                                    break
                    else:
                        path = get_path_to_targets_meters(scene_graph, target_labels, SCALA_PIXEL_METRI)
                    if path:
                        print("\nPATH VERSO GLI OBIETTIVI:")
                        for step, target in enumerate(path):
                            print(f"  Step {step+1}: {target['label']} → pixel: {target['position_pixel']} | metri: {target['position_metri']}")
                    else:
                        print("Nessun oggetto trovato.")

                elif cmd == 'g':
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    descrizione = input("Descrivi il target a parole tue: ")
                    scene_json = scene_graph_to_json(scene_graph, FREQ_THRESHOLD)
                    target_labels = resolve_targets(descrizione, scene_json)
                    if not target_labels:
                        print("Gemini non ha trovato oggetti corrispondenti nella scena.")
                    else:
                        print(f"Gemini ha identificato: {target_labels}")

                        # Il planning gestisce un solo target alla volta: il primo diventa "attivo" e
                        # viene ripubblicato automaticamente ad ogni frame (vedi republish_active_target).
                        # Gli altri restano solo nel path stampato, in attesa della gestione multi-goal.
                        self.active_target_label = target_labels[0]

                        path = []

                        for label in target_labels:
                            for node in scene_graph:
                                if node['label'] == label and node['count'] >= FREQ_THRESHOLD:
                                    pos = node.get('position')
                                    if pos and current_z is not None:
                                        pos_metri = pixel_to_meters_3d(pos[0], pos[1], current_z, CAMERA_MATRIX)
                                        path.append({'label': label, 'position_pixel': pos, 'position_metri': pos_metri})
                                    elif pos:
                                        pos_metri = (pos[0] * SCALA_PIXEL_METRI, pos[1] * SCALA_PIXEL_METRI)
                                        path.append({'label': label, 'position_pixel': pos, 'position_metri': pos_metri})
                                    break
                        if path:
                            print("\nPATH VERSO GLI OBIETTIVI (da linguaggio naturale):")
                            for step, t in enumerate(path):
                                print(f"  Step {step+1}: {t['label']} → pixel: {t['position_pixel']} | metri: {t['position_metri']}")
                        else:
                            print("Oggetti identificati ma posizione non disponibile.")
                
                elif cmd == 'v':
                    # Verifica posizioni: distanze a coppie + posizione relativa al marker ArUco
                    import itertools
                    oggetti = []
                    for node in scene_graph:
                        if node['count'] >= FREQ_THRESHOLD and node.get('position'):
                            pos = node['position']
                            if current_z is not None:
                                pm = pixel_to_meters_3d(pos[0], pos[1], current_z, CAMERA_MATRIX)
                            else:
                                pm = (pos[0] * SCALA_PIXEL_METRI, pos[1] * SCALA_PIXEL_METRI)
                            oggetti.append({'label': node['label'], 'pixel': pos, 'metri': pm})

                    if len(oggetti) < 1:
                        print("Nessun oggetto stabile nella scena.")
                    else:
                        print("\n" + "="*50)
                        print("VERIFICA POSIZIONI")
                        print("="*50)
                        for o in oggetti:
                            print(f"  {o['label']}: pixel {o['pixel']} | metri ({o['metri'][0]:.4f}, {o['metri'][1]:.4f})")

                        # Test 1 — distanze a coppie
                        if len(oggetti) >= 2:
                            print("\n  DISTANZE A COPPIE (test di coerenza):")
                            for a, b in itertools.combinations(oggetti, 2):
                                dx = b['metri'][0] - a['metri'][0]
                                dy = b['metri'][1] - a['metri'][1]
                                dist = (dx**2 + dy**2) ** 0.5
                                print(f"    {a['label']} <-> {b['label']}: {dist:.4f} m ({dist*100:.1f} cm)")

                        # Test 2 — posizione relativa al marker ArUco
                        if current_z is not None:
                            print(f"\n  z corrente (ArUco): {current_z:.4f} m")
                            print("  (per il test assoluto: confronta gli spostamenti relativi tra oggetti")
                            print("   con le distanze misurate fisicamente sul piano)")
                        print("="*50 + "\n")

                elif cmd == 'q':
                    if rclpy.ok():
                        rclpy.shutdown()
                    break
            except EOFError:
                break

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
        print_scene_graph()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()