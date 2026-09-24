from cProfile import label

import torch
import clip
import cv2
import numpy as np
from PIL import Image
import sys
import os
import threading
import itertools
import time
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Bool, Float32MultiArray

# ROS2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.onnx_model import SGG_ONNX_Model
from demo.query_scene import get_position_history, get_current_position, get_path_to_targets_meters
from demo.aruco_detector import detect_aruco, pixel_to_meters_3d, CAMERA_MATRIX, calcola_scala_da_aruco
from demo.graspnet_node import BBOX_MARGIN_M

# ── Configurazione ──────────────────────────────────────────
ONNX_PATH = "checkpoints/VG150/react++_yolov8m/model.onnx"
SIMILARITY_THRESHOLD = 0.85
FREQ_THRESHOLD = 2
# Soglia sotto la quale il nodo viene rimosso dal grafo (oggetto considerato "perso" davvero).
CONFIDENCE_REMOVE_THRESHOLD = 0.05

# Decadimento basato sul TEMPO REALE trascorso (23/09/2026), non piu' su un
# fattore fisso per ciclo elaborato -- con SGG a velocita' variabile (oggi tra
# ~1.5s e ~5s a ciclo) un decay per-frame fa durare la "memoria" in secondi
# reali in modo incoerente (misurato: 4-12+ minuti prima della rimozione,
# troppo). Stesso principio gia' usato per il filtro EMA in ET_node.cpp
# (master_direction, coefficiente da un dt reale, non fisso).
# MEMORY_SECONDS_TARGET: quanti secondi reali senza essere rivisto prima che
# un nodo scenda sotto CONFIDENCE_REMOVE_THRESHOLD, partendo da confidence=1.0.
# Valore di partenza non misurato/concordato, da tarare in laboratorio.
MEMORY_SECONDS_TARGET = 30.0

# Stadio B: cicli consecutivi in cui lo stesso oggetto deve essere l'unico
# visto prima della selezione automatica.
AUTO_SELECT_STABLE_CYCLES = 3
CONFIDENCE_DECAY_PER_SECOND = CONFIDENCE_REMOVE_THRESHOLD ** (1.0 / MEMORY_SECONDS_TARGET)
SCALA_PIXEL_METRI = 0.000435  # fallback se ArUco non visibile, calcolato con z=0.397m

DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
DEPTH_WINDOW = 5  # finestra NxN attorno al pixel target per mediare il depth e ridurre il rumore
# Sotto questa distanza la D435 non e' affidabile (range minimo misurato in lab
# il 14/09: ~20.4 cm). Letture piu' vicine vengono scartate: il 18/09 una lettura
# a 0.218 m ha spostato il target di oltre 10 cm in un ciclo, facendo scattare il
# braccio all'indietro.
MIN_VALID_DEPTH_M = 0.25

# Margine dal bordo dell'immagine (in pixel) entro cui un oggetto è considerato
# "vicino al bordo" — se l'ultima posizione nota di un nodo era in questa fascia,
# è più plausibile che sia appena uscito dal campo visivo (camera eye-in-hand che
# si è spostata) piuttosto che effettivamente sparito dalla scena.
EDGE_MARGIN_PX = 40

BLACKLIST_OBJECTS = {
    'floor', 'wall', 'ceiling', 'chair',
    'ground', 'background', 'window', 'door',
    'hair', 'nose', 'table','face', 'head', 'eye', 'ear',
    'mouth', 'neck', 'arm', 'leg',
    # Aggiunti il 23/09/2026: il banco da laboratorio viene classificato dal
    # rilevatore come "counter" o "sink" (classi VG150), non come "table" --
    # senza queste due, lo sfondo entrava tra i candidati dello Stadio B
    # (selezione automatica a un solo oggetto), che non scattava mai perche'
    # non c'era mai un solo candidato.
    'counter', 'sink'
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
# Tracker OC-SORT attivabile senza modificare il file: SGG_TRACKING=1 python demo/sgg_ros_node.py
TRACKING = os.environ.get('SGG_TRACKING', '0') == '1'
print(f"Tracking OC-SORT: {'attivo' if TRACKING else 'disattivo'}")
sgg = SGG_ONNX_Model(None, ONNX_PATH, tracking=TRACKING)

print("Carico CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"Pronti! Uso: {device}")

# ── Albero semantico ────────────────────────────────────────
# scene_graph e' mutato dentro frame_callback SOTTO self.lock (vedi
# SGGNode.frame_callback). L'unica vera race era con command_loop, che gira su
# un threading.Thread separato dal contesto rclpy: un decay/pop concorrente in
# frame_callback mentre un handler (p/h/t/g/v) stava iterando scene_graph
# poteva leggere stato inconsistente o sollevare un'eccezione. Corretto
# (12/09): ogni handler ora prende uno snapshot (list(scene_graph)) sotto
# self.lock subito dopo l'ultimo input()/chiamata di rete bloccante, e lavora
# sulla copia — il lock non resta mai tenuto durante un'attesa da tastiera o
# una chiamata a Gemini, altrimenti bloccherebbe frame_callback per tutto quel
# tempo. gripper_status_callback -> advance_to_next_object, nonostante letto in
# precedenza come a rischio, NON e' in race: main() usa rclpy.spin() a singolo
# thread, quindi gira sempre serializzato rispetto a frame_callback (nessuna
# concorrenza reale tra due callback rclpy in questo nodo, solo tra i callback
# rclpy nel loro insieme e command_loop).
scene_graph = []
current_z = None  # aggiornata quando viene rilevato un marker ArUco
_last_decay_time = None  # tempo reale (time.time()) dell'ultimo decadimento applicato, per il dt in update_scene_graph

# Identificatore stabile di ogni nodo: le relazioni lo usano al posto
# dell'indice nella lista, che cambia a ogni scene_graph.pop().
_next_node_uid = itertools.count()


def node_by_uid(sg, uid):
    return next((n for n in sg if n.get('uid') == uid), None)

# ── Funzioni ────────────────────────────────────────────────

def imgmsg_to_numpy_bgr8(msg):
    """Sostituisce cv_bridge.imgmsg_to_cv2(msg, 'bgr8') con una conversione
    manuale — evita la dipendenza da cv_bridge (compilato contro NumPy 1.x
    dal sistema ROS2), permettendo a boxmot (che richiede NumPy 2.x) di
    coesistere senza conflitto.

    Il topic REALE (/camera/camera/color/image_raw) pubblica in encoding
    'rgb8', verificato dal vivo il 21/09/2026 (`ros2 topic echo --field
    encoding` -> rgb8), non 'bgr8': questa funzione fa un reshape puro senza
    scambiare i canali, quindi il colore restituito e' in realta' RGB anche
    se il nome fa pensare a BGR. Un tentativo di correggere lo scambio (21/09,
    poi 22/09) era corretto sulla carta (confermato anche in tools/export_onnx.py,
    che usa davvero cv2.imread + BGR->RGB) ma ha peggiorato l'identificazione
    nei test reali in laboratorio -- riportato al comportamento originale il
    22/09/2026 sulla base del riscontro pratico, prevale sul ragionamento
    teorico. Ipotesi non confermata: VG150 e' sbilanciato verso poche classi
    frequenti (es. \"man\"), e con i colori teoricamente corretti il modello
    puo' diventare piu' incerto su alcuni oggetti e scivolare su quelle classi."""
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)


def imgmsg_to_numpy_depth16(msg):
    """Converte un sensor_msgs/Image in formato 16UC1 (depth allineato della
    RealSense, valori in millimetri) in un array numpy — stessa logica di
    imgmsg_to_numpy_bgr8, evita cv_bridge."""
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)


def read_depth_at_pixel(depth_frame, cx_pixel, cy_pixel, window=DEPTH_WINDOW):
    """Legge la profondita' reale (in metri) mediando una finestra NxN attorno
    al pixel dato — un solo pixel sarebbe troppo rumoroso. Scarta gli zeri
    (letture invalide/mancanti della RealSense) e usa la mediana, piu' robusta
    della media in presenza di outlier. Restituisce None se nella finestra non
    c'e' nessuna lettura valida (es. troppo vicino al limite operativo del
    sensore, tipicamente ~20cm)."""
    h, w = depth_frame.shape
    half = window // 2
    x0, x1 = max(0, int(cx_pixel) - half), min(w, int(cx_pixel) + half + 1)
    y0, y1 = max(0, int(cy_pixel) - half), min(h, int(cy_pixel) + half + 1)
    patch = depth_frame[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    depth_mm = np.median(valid)
    return depth_mm / 1000.0  # mm -> metri (formato Z16 standard RealSense)


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

def compute_iou(box1, box2):
    """Intersection over Union tra due bounding box (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


IOU_THRESHOLD = 0.3  # sopra questo valore, consideriamo i box "nella stessa posizione"

def find_existing_node(label, embedding, bbox,track_id=None):
    for i, node in enumerate(scene_graph):
        if node['label'] == label:
            # Criterio più forte: stesso ID di traccia dato dal tracker
            # (OC-SORT/ByteTrack, attivo solo se tracking=true). Basato sul
            # moto stimato, più robusto di CLIP/IoU quando cambiano luce o
            # angolo — se combacia, ci fidiamo subito.
            if track_id is not None and node.get('track_id') == track_id:
                return i


            sim = torch.cosine_similarity(node['embedding'], embedding).item()
            iou = compute_iou(node['bbox'], bbox)
            if sim >= SIMILARITY_THRESHOLD or iou >= IOU_THRESHOLD:
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

        x1, y1, x2, y2 = map(int, box[:4])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # L'ID di traccia (colonna 6) esiste solo se tracking=true è stato
        # passato a SGG_ONNX_Model — altrimenti box ha solo le colonne
        # originali (coordinate, score, label) e questo resta None.
        # Il tracker assegna id >= 1; 0 e' il valore di riempimento per i box
        # senza traccia e non deve far combaciare oggetti diversi.
        track_id = int(box[6]) if len(box) > 6 and int(box[6]) > 0 else None

        existing_idx = find_existing_node(label, embedding, (x1, y1, x2, y2),track_id)

        if existing_idx >= 0:
            scene_graph[existing_idx]['count'] += 1
            scene_graph[existing_idx]['embedding'] = embedding
            scene_graph[existing_idx]['position'] = (cx, cy)
            scene_graph[existing_idx]['bbox'] = (x1, y1, x2, y2)
            scene_graph[existing_idx]['position_history'].append((cx, cy))
            scene_graph[existing_idx]['position_history'] = scene_graph[existing_idx]['position_history'][-50:]
            scene_graph[existing_idx]['frames_not_seen'] = 0
            scene_graph[existing_idx]['confidence'] = 1.0          # <-- piena confidenza quando rivisto
            scene_graph[existing_idx]['track_id'] = track_id
            box_to_node[i] = existing_idx
        else:
            new_node = {
                'uid': next(_next_node_uid),
                'label': label,
                'embedding': embedding,
                'relazioni': {},
                'count': 1,
                'position': (cx, cy),
                'bbox': (x1, y1, x2, y2),
                'position_history': [(cx, cy)],
                'frames_not_seen': 0,
                'confidence': 1.0,
                'track_id': track_id  
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

                rel_key = (pred, scene_graph[obj_node]['uid'])
                if rel_key in scene_graph[subj_node]['relazioni']:
                    scene_graph[subj_node]['relazioni'][rel_key] += 1
                else:
                    scene_graph[subj_node]['relazioni'][rel_key] = 1


    # TODO (estensione futura): PERCHÉ il nodo non è stato visto in questo ciclo. 
    #   (a) fuori dal field of view della camera (la camera si è spostata, eye-in-hand),
    #   (b) occluso da un ostacolo o dal gripper durante il grasping.
    # Per (a): conosciamo la posa nota della camera via TF
    # (base_link -> camera_color_optical_frame); si potrebbe proiettare la
    # posizione 3D nota del nodo nel frame camera corrente e controllare se cade
    # dentro i limiti dell'immagine — se fuori, congelare il decay per questo nodo.
    # Per (b): servirebbe sapere se il gripper sta transitando sopra la posizione
    # nota del nodo (serve la posa del gripper, non solo la camera).

    global _last_decay_time
    now = time.time()
    dt = (now - _last_decay_time) if _last_decay_time is not None else 0.0
    _last_decay_time = now
    # dt<=0 (primo giro, o orologio non avanzato) -> nessun decadimento questo
    # ciclo, stesso principio guardia usato per il filtro EMA in ET_node.cpp.
    decay_factor = (CONFIDENCE_DECAY_PER_SECOND ** dt) if dt > 0.0 else 1.0

    seen_nodes = set(box_to_node.values())
    to_remove = []
    for i, node in enumerate(scene_graph):
        if i not in seen_nodes:
            node['frames_not_seen'] += 1

            # Euristica semplificata per il campo visivo. Qui usiamo solo la
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
                node['confidence'] *= decay_factor

            if node['confidence'] < CONFIDENCE_REMOVE_THRESHOLD:
                print(f"  ⚠️ '{node['label']}' sparito dalla scena — probabilmente preso dal robot.")
                to_remove.append(i)

    for i in sorted(to_remove, reverse=True):
        scene_graph.pop(i)

def print_scene_graph(sg=None):
    """sg: snapshot opzionale di scene_graph (vedi command_loop, che ne prende
    uno sotto lock prima di chiamare questa funzione da un thread diverso da
    quello rclpy). None = usa la globale direttamente, sicuro solo se non c'e'
    concorrenza (es. allo shutdown in main())."""
    if sg is None:
        sg = scene_graph
    print("\n" + "="*50)
    print(f"ALBERO SEMANTICO — {len(sg)} oggetti nella scena")
    print("="*50)
    for i, node in enumerate(sg):
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
        tid = f" — track_id: {node['track_id']}" if node.get('track_id') is not None else ""
        print(f"  [{i}] {node['label']} (visto {node['count']} volte) — storia: {len(history)} punti{tid}")
        print(f"       pos pixel: {pos} | pos metri: ({pos_metri[0]:.3f}m, {pos_metri[1]:.3f}m) | {depth_info}")
        if bbox:
            print(f"       bbox: {bbox} — larghezza: {bbox[2]-bbox[0]}px, altezza: {bbox[3]-bbox[1]}px")
        for (pred, obj_uid), count in list(node['relazioni'].items()):
            if count < FREQ_THRESHOLD:
                continue
            if pred not in VG_TO_FUNCTIONAL and pred not in VG_TO_SPATIAL:
                continue
            obj = node_by_uid(sg, obj_uid)
            if obj is None:
                continue
            print(f"       --({pred})--> {obj['label']} [vista {count}x]")
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
        self.frame_count = 0
        self.img = None
        self.lock = threading.Lock()
        self.active_target_label = None   # <--label del target da ripubblicare ad ogni frame

        # Stadio B v2 (23/09/2026 sera): selezione automatica del target
        # senza comando esplicito -- SOLO lato percezione, mai tocca
        # ET_node.cpp ne' il canale multi-candidato (/sgg/candidate_targets).
        # Se attivo e non c'e' un comando esplicito, quando in scena resta
        # esattamente un candidato ci si comporta come se fosse arrivato un
        # `t` su di lui: si riusa per intero il canale a target singolo
        # (/sgg/target_point) gia' validato -- stesso congelamento, stessa
        # media letture, stessa soglia di distanza lato ET_node (usare
        # trigger_distance piu' basso, es. 0.10, come parametro di lancio
        # di ET_node per questo caso, non qui). Off di default. Sostituisce
        # il primo tentativo (parametro in ET_node.cpp, rimosso il 23/09
        # dopo un incidente: il canale multi-candidato porta con se' un
        # inseguimento continuo fuori zona di grasp, non voluto).
        self.declare_parameter('auto_select_single_target', False)
        self.auto_select_single_target = self.get_parameter('auto_select_single_target').get_parameter_value().bool_value
        self._auto_select_last_label = None
        self._auto_select_streak = 0

        # Arbitro del target, Stadio C (24/09/2026). 'off' (default) oppure
        # 'ombra': calcola e scrive nel log cosa sceglierebbe fra gli oggetti
        # visti, SENZA pubblicare nulla -- il robot si comporta come negli
        # stadi A/B. Serve a vedere sui dati reali se le decisioni hanno senso
        # prima di dargli il controllo. Vedi demo/target_arbiter.py.
        self.declare_parameter('arbiter_mode', 'off')
        self.arbiter_mode = self.get_parameter('arbiter_mode').get_parameter_value().string_value
        self.arbiter = None
        if self.arbiter_mode not in ('off', 'ombra'):
            self.get_logger().warn(
                f"arbiter_mode='{self.arbiter_mode}' non supportato (solo 'off' o 'ombra'): arbitro spento.")
            self.arbiter_mode = 'off'
        if self.arbiter_mode == 'ombra':
            self._setup_arbiter()

        # Limita la FREQUENZA DI STAMPA (non di pubblicazione, quella resta
        # a ogni frame) di "Target pubblicato"/"Candidati pubblicati": senza
        # questo, a 15-30 Hz il terminale stampa piu' veloce di quanto si
        # riesca a scrivere un comando in input() (girano su thread diversi,
        # command_loop vs il callback camera) -- aggiunto il 21/09/2026 dopo
        # che in lab era difficile scrivere "bottle" tra uno stampa e l'altra.
        self._print_throttle_s = 1.0
        self._last_target_print = 0.0
        self._last_candidates_print = 0.0
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

        self.last_depth_frame = None
        self.depth_sub = self.create_subscription(
            RosImage,
            DEPTH_TOPIC,
            self.depth_callback,
            10)

        # Publisher della posizione del target verso il nodo MoveIt
        self.target_pub = self.create_publisher(PointStamped, '/sgg/target_point', 10)

        # Bbox pixel [x1, y1, x2, y2] del target, verso graspnet_node.py
        # (22/09/2026): stesso identico refuso-a-runtime possibile di
        # candidates_pub/target_pub, il nome del topic deve combaciare con
        # target_bbox_topic dichiarato in graspnet_node.py.
        self.target_bbox_pub = self.create_publisher(Float32MultiArray, '/sgg/target_bbox', 10)

        # Timer per visualizzazione (ogni 100ms)
        self.create_timer(0.1, self.display_callback)

        # Thread separato per i comandi da tastiera
        self.cmd_thread = threading.Thread(target=self.command_loop, daemon=True)
        self.cmd_thread.start()

        self.get_logger().info("SGG Node avviato — in ascolto su /camera/camera/color/image_raw")

    def depth_callback(self, msg):
        self.last_depth_frame = imgmsg_to_numpy_depth16(msg)

    def frame_callback(self, msg):
        frame = imgmsg_to_numpy_bgr8(msg)
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
                global current_z, SCALA_PIXEL_METRI
                if aruco_results:
                    current_z = aruco_results[0]['z']
                    SCALA_PIXEL_METRI = calcola_scala_da_aruco(aruco_results[0]['corners'])
                    # Calibrazione: distanza tra due marker
                    if len(aruco_results) >= 2:
                        p1 = aruco_results[0]['position']
                        p2 = aruco_results[1]['position']
                        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                        print(f"  📏 Distanza tra marker: {dist:.3f}m (attesa: 0.20m)")
                elif self.last_depth_frame is not None:
                    # ArUco non visibile (probabilmente fuori campo per avvicinamento
                    # ravvicinato) — fallback sulla depth reale della RealSense, letta
                    # nel pixel del target attivo. A differenza del piano comune
                    # dato da ArUco, qui la profondita' e' quella vera dell'oggetto
                    # target, non un'approssimazione condivisa per tutta la scena.
                    #
                    # Due casi in cui sappiamo SENZA ambiguita' quale pixel campionare:
                    # 1) target attivo esplicito (comando t/g o avanzamento post-grasp);
                    # 2) nessun comando esplicito, ma get_candidate_targets() e' sceso a
                    #    un solo candidato — stesso segnale gia' usato da
                    #    republish_candidate_targets() qui sotto, quindi nessuna logica
                    #    nuova, solo riuso. Con 2+ candidati restiamo davvero ambigui
                    #    (non sapremmo su quale pixel campionare: ET_node sceglie il
                    #    vincitore per allineamento col comando master, un'informazione
                    #    che qui non abbiamo) e non applichiamo il fallback — current_z
                    #    resta quello che era, comportamento invariato in quel caso.
                    pos_pixel = None
                    if self.active_target_label is not None:
                        target_node = next((n for n in scene_graph if n['label'] == self.active_target_label), None)
                        if target_node is not None and target_node.get('position') is not None:
                            pos_pixel = target_node['position']
                    else:
                        candidates = get_candidate_targets()
                        if len(candidates) == 1:
                            pos_pixel = candidates[0]['position']

                    if pos_pixel is not None:
                        depth_m = read_depth_at_pixel(self.last_depth_frame, pos_pixel[0], pos_pixel[1])
                        if depth_m is not None and depth_m < MIN_VALID_DEPTH_M:
                            print(f"  ⚠️ Fallback depth: lettura {depth_m:.3f} m sotto il minimo affidabile "
                                  f"({MIN_VALID_DEPTH_M} m) — z invariata")
                            depth_m = None
                        if depth_m is not None:
                            current_z = depth_m
                            print(f"  📏 Fallback depth attivo: z={depth_m:.3f}m (pixel {pos_pixel})")
                        else:
                            print(f"  ⚠️ Fallback depth: nessuna lettura valida nella finestra attorno a {pos_pixel} (troppo vicino/fuori range?) — z invariata")
                    else:
                        print("  ⚠️ Fallback depth: ArUco non visibile ma nessun pixel target disponibile (target ambiguo o non in scena) — z invariata")

                # Stadio B v2: selezione automatica, solo se nessun comando
                # esplicito e resta esattamente un candidato -- si comporta
                # come se fosse arrivato un `t`, poi tutto il resto (riga
                # sotto) segue il normale canale a target singolo.
                # dbg is None: grafo non aggiornato in questo ciclo, non far
                # avanzare la conferma su dati vecchi.
                if self.active_target_label is None and self.auto_select_single_target and dbg is not None:
                    # Solo oggetti visti in QUESTO ciclo: i nodi in memoria
                    # (frames_not_seen > 0) non devono contare come secondo
                    # oggetto in scena -- il 23/09 un 'cap' in memoria ha
                    # bloccato la selezione per ~75s dopo essere sparito.
                    auto_candidates = [
                        n for n in scene_graph
                        if n['count'] >= FREQ_THRESHOLD
                        and n.get('confidence', 0.0) >= CONFIDENCE_REMOVE_THRESHOLD
                        and n.get('frames_not_seen', 0) == 0
                    ]
                    # Conferma su piu' cicli: con lo sfarfallio del rilevatore
                    # un singolo ciclo "pulito" potrebbe mostrare solo una
                    # parte dell'oggetto (es. 'cap' senza 'bottle').
                    single = auto_candidates[0]['label'] if len(auto_candidates) == 1 else None
                    if single is not None and single == self._auto_select_last_label:
                        self._auto_select_streak += 1
                    else:
                        self._auto_select_streak = 1 if single is not None else 0
                    self._auto_select_last_label = single
                    if single is not None and self._auto_select_streak >= AUTO_SELECT_STABLE_CYCLES:
                        self.active_target_label = single
                        self._auto_select_streak = 0
                        self._auto_select_last_label = None
                        print(f"  🤖 Selezione automatica (Stadio B): target impostato su '{self.active_target_label}' "
                              f"(unico oggetto visto per {AUTO_SELECT_STABLE_CYCLES} cicli consecutivi).")

                # Arbitro in modalita' ombra: solo log, non tocca il target.
                if self.arbiter is not None and dbg is not None:
                    self._arbiter_shadow_step()

                self.republish_active_target()   # ripubblica il target attivo, se c'è
                self.republish_candidate_targets()   # candidati multipli, solo se nessun comando esplicito attivo

    def display_callback(self):
        with self.lock:
            if self.img is not None:
                # Disegna sopra l'immagine (già renderizzata dalla libreria SGG)
                # anche i nodi con confidenza sufficiente ma non rilevati in
                # QUESTO frame — usa l'ultima posizione/bbox nota in scene_graph.
                # Serve a mantenere il riquadro visibile durante lo sfarfallio
                # del rilevatore (oggetto reale ancora presente, solo non
                # rilevato in questo specifico frame).
                display_img = self.img.copy()
                for node in scene_graph:
                    if node.get('frames_not_seen', 0) > 0 and node.get('confidence', 0.0) >= CONFIDENCE_REMOVE_THRESHOLD:
                        bbox = node.get('bbox')
                        if bbox:
                            x1, y1, x2, y2 = bbox
                            # Colore diverso (arancione) per distinguere visivamente
                            # i nodi "persistenti" da quelli rilevati in questo frame
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(display_img, f"{node['label']} (memoria)", (x1, max(y1 - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Debug (22/09/2026): disegna il bbox del target attivo e il
                # crop con margine che riceverebbe grasp_server.py, per
                # verificare a occhio l'allineamento — vedi BBOX_MARGIN_M in
                # graspnet_node.py.
                if self.active_target_label is not None:
                    target_node = next((n for n in scene_graph if n['label'] == self.active_target_label), None)
                    if target_node is not None:
                        bbox = target_node.get('bbox')
                        pos = target_node.get('position')
                        if bbox is not None and pos is not None:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            z, _ = self.z_for_pixel(pos)
                            if z:
                                fx, fy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1]
                                mx, my = int(BBOX_MARGIN_M * fx / z), int(BBOX_MARGIN_M * fy / z)
                                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                                cx2, cy2 = x2 + mx, y2 + my
                                cv2.rectangle(display_img, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
                                cv2.putText(display_img, "crop GraspNet", (cx1, max(cy1 - 8, 0)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                # Fix 22/09/2026: mostrava self.img (senza i riquadri appena
                # disegnati sopra) invece di display_img -- i riquadri
                # "memoria" e quelli di debug del target non comparivano mai.
                cv2.imshow("SGG ROS2 Node", display_img)
                cv2.waitKey(1)
    
    # ── Arbitro del target, modalita' ombra ────────────────────
    def _setup_arbiter(self):
        # Import qui: con arbitro spento il nodo non dipende da TF.
        import tf2_ros
        from rclpy.time import Time
        from geometry_msgs.msg import PoseStamped
        from tf2_geometry_msgs import do_transform_point
        from demo.target_arbiter import TargetArbiter

        self.arbiter = TargetArbiter()
        self._tf_time_latest = Time()
        self._do_transform_point = do_transform_point
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._ee_pos = None
        self._master_pos = None
        self._pose_frames_logged = set()
        self._last_arbiter_print = 0.0
        # TODO (audit): letterali da tenere allineati ai default pose_topic e
        # desired_pose_topic dichiarati in ET_node.cpp (repo Energy-Tanks).
        self.create_subscription(PoseStamped, '/admittance_controller/pose_debug',
                                 lambda m: self._store_pose(m, '_ee_pos'), 10)
        self.create_subscription(PoseStamped, '/twist_to_pose_converter/desired_pose',
                                 lambda m: self._store_pose(m, '_master_pos'), 10)
        self.get_logger().info("Arbitro in modalita' OMBRA: calcola e scrive nel log, non pubblica nulla.")

    def _store_pose(self, msg, attr):
        setattr(self, attr, np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]))
        # Il disegno assume entrambe le pose in base_link: lo si verifica qui,
        # una volta per topic.
        if attr not in self._pose_frames_logged:
            self._pose_frames_logged.add(attr)
            print(f"  [arbitro ombra] {attr}: frame_id='{msg.header.frame_id}'")

    def _arbiter_shadow_step(self):
        """Un ciclo dell'arbitro sugli oggetti visti ora: solo log."""
        try:
            tf = self._tf_buffer.lookup_transform('base_link', 'camera_color_optical_frame',
                                                  self._tf_time_latest)
        except Exception as exc:
            self._arbiter_print(f"TF camera->base_link non disponibile ({type(exc).__name__}): nessuna decisione.")
            return

        candidates = []
        for n in scene_graph:
            if (n['count'] < FREQ_THRESHOLD or n.get('confidence', 0.0) < CONFIDENCE_REMOVE_THRESHOLD
                    or n.get('frames_not_seen', 0) != 0):
                continue
            pos, bbox = n.get('position'), n.get('bbox')
            if pos is None or bbox is None:
                continue
            z, _ = self.z_for_pixel(pos, quiet=True)
            if z is None:
                continue
            pm = pixel_to_meters_3d(pos[0], pos[1], z, CAMERA_MATRIX)
            p = PointStamped()
            p.header.frame_id = 'camera_color_optical_frame'
            p.point.x, p.point.y, p.point.z = float(pm[0]), float(pm[1]), float(z)
            pb = self._do_transform_point(p, tf).point
            candidates.append({'uid': n['uid'], 'label': n['label'], 'bbox': bbox,
                               'position_base': (pb.x, pb.y, pb.z)})

        if self._ee_pos is None:
            raw_dir, ee, posa = None, np.zeros(3), "posa robot assente"
        elif self._master_pos is None:
            raw_dir, ee, posa = None, self._ee_pos, "comando Falcon assente"
        else:
            raw_dir, ee, posa = self._master_pos - self._ee_pos, self._ee_pos, "ok"

        res = self.arbiter.step(candidates, raw_dir, ee, time.time())

        names = {c['uid']: f"{c['label']}#{c['uid']}" for c in candidates}
        names_all = {n['uid']: f"{n['label']}#{n['uid']}" for n in scene_graph}
        cand_txt = ", ".join(names[c['uid']] for c in candidates) or "nessuno"
        score_txt = " ".join(f"{names[u]}={s:.2f}" for u, s in res.scores.items()) or "-"
        prop_txt = names_all.get(res.proposal_uid, "nessuna") if res.proposal_uid is not None else "nessuna"
        scelta_txt = names_all.get(res.target_uid, f"#{res.target_uid}") if res.target_uid is not None else "nessuna"
        self._arbiter_print(
            f"candidati: {cand_txt} | punteggi: {score_txt} | dir={res.direction_norm:.3f} m ({posa}) | "
            f"proposta: {prop_txt} | scelta: {scelta_txt} ({res.reason}) | "
            f"target esplicito: {self.active_target_label or 'nessuno'}")

    def _arbiter_print(self, text):
        now = time.time()
        if now - self._last_arbiter_print >= self._print_throttle_s:
            self._last_arbiter_print = now
            print(f"  [arbitro ombra] [{now:.3f}] {text}")

    def z_for_pixel(self, pos_pixel, quiet=False):
        """Profondita' dell'oggetto in quel pixel: depth reale della RealSense
        se la lettura e' valida, altrimenti il piano ArUco (current_z).
        Il piano ArUco da solo sbaglia la z degli oggetti alti (una bottiglia
        ha la cima piu' vicina alla camera del marker sul tavolo).
        Restituisce (z, sorgente) oppure (None, None)."""
        if self.last_depth_frame is not None:
            depth_m = read_depth_at_pixel(self.last_depth_frame, pos_pixel[0], pos_pixel[1])
            if depth_m is not None and depth_m >= MIN_VALID_DEPTH_M:
                return depth_m, 'depth'
            if depth_m is not None and not quiet:
                print(f"  ⚠️ Depth {depth_m:.3f} m sotto il minimo affidabile "
                      f"({MIN_VALID_DEPTH_M} m) nel pixel {pos_pixel}: lettura scartata.")
        if current_z is not None:
            return current_z, 'aruco'
        return None, None

    def pubblica_target(self, pos_pixel, bbox=None):
        """Pubblica la posizione 3D del target nel frame camera, e
        opzionalmente il suo bbox pixel su /sgg/target_bbox (22/09/2026,
        usato da graspnet_node.py per il crop X/Y lato server)."""
        z, sorgente = self.z_for_pixel(pos_pixel)
        if z is None:
            self.get_logger().warn("Nessuna profondita' disponibile (ne' depth ne' ArUco): non pubblico il target.")
            return
        pm = pixel_to_meters_3d(pos_pixel[0], pos_pixel[1], z, CAMERA_MATRIX)
        msg = PointStamped()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(pm[0])
        msg.point.y = float(pm[1])
        msg.point.z = float(z)
        self.target_pub.publish(msg)
        if bbox is not None:
            bbox_msg = Float32MultiArray()
            bbox_msg.data = [float(v) for v in bbox]
            self.target_bbox_pub.publish(bbox_msg)
        now = time.time()
        # Timestamp epoch (stesso formato dei log di ET_node, es. [1790001869.839])
        # per poter allineare a occhio quando serve, aggiunto il 21/09/2026.
        if now - self._last_target_print >= self._print_throttle_s:
            self._last_target_print = now
            print(f"  → [{now:.3f}] Target pubblicato su /sgg/target_point: ({pm[0]:.3f}, {pm[1]:.3f}, {z:.3f}) [frame camera, z da {sorgente}]")

    def gripper_status_callback(self, msg: Bool):
        """Ogni messaggio su questo topic è già un grasp confermato (ET_node ha
        già verificato transizione + distanza) — nessuna logica di transizione
        necessaria qui."""
        if self.arbiter is not None:
            self.arbiter.reset()   # la scelta dell'arbitro vale fino al grasp
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
        for (pred, obj_uid), count in grasped_node['relazioni'].items():
            if pred in VG_TO_FUNCTIONAL and count > best_count and node_by_uid(scene_graph, obj_uid) is not None:
                best_rel, best_count = (pred, obj_uid), count

        if best_rel is None:
            print(f"  ℹ️ Nessuna relazione funzionale per '{grasped_label}': nessun successivo.")
            self.active_target_label = None
            return

        pred, obj_uid = best_rel
        candidate_label = node_by_uid(scene_graph, obj_uid)['label']

        # Giudice in sola modalita' log: un suo errore (chiave mancante, rete)
        # non deve fermare il nodo.
        try:
            from demo.gemini_retrieval import judge_functional_relation
            verdict = judge_functional_relation(grasped_label, pred, candidate_label)
        except Exception as exc:
            print(f"  🤖 Giudice LLM non disponibile ({exc}).")
            verdict = None
        # TODO: modalità SOLO LOG per validare il giudice prima di fidarsene --
        # non cambia ancora il comportamento (active_target_label si imposta
        # comunque come oggi, indipendentemente dal verdetto). Prossimo passo,
        # dopo verifica in lab: se verdict['plausible'] è False, scartare
        # questo candidato e riprovare col prossimo per count invece di
        # fermarsi; se Gemini non risponde (verdict is None), comportamento
        # invariato (nessun veto, si va avanti come oggi).
        if verdict is not None:
            esito = "APPROVATO" if verdict['plausible'] else "BOCCIATO"
            print(f"  🤖 Giudice LLM: {esito} (score={verdict['score']:.2f}) — {verdict['reason']}")
        else:
            print("  🤖 Giudice LLM non raggiungibile, nessun veto (solo log per ora).")

        self.active_target_label = candidate_label
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
                        self.pubblica_target(pos, bbox=node.get('bbox'))
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

        candidates = get_candidate_targets()
        if not candidates:
            return

        msg = PoseArray()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.header.stamp = self.get_clock().now().to_msg()

        sorgenti = []
        for cand in candidates:
            pos_pixel = cand['position']
            z, sorgente = self.z_for_pixel(pos_pixel)
            if z is None:
                continue
            pm = pixel_to_meters_3d(pos_pixel[0], pos_pixel[1], z, CAMERA_MATRIX)
            pose = Pose()
            pose.position.x = float(pm[0])
            pose.position.y = float(pm[1])
            pose.position.z = float(z)
            pose.orientation.w = 1.0
            msg.poses.append(pose)
            sorgenti.append(f"{cand['label']}:z={z:.3f}({sorgente})")

        if not msg.poses:
            return
        self.candidates_pub.publish(msg)
        now = time.time()
        if now - self._last_candidates_print >= self._print_throttle_s:
            self._last_candidates_print = now
            print(f"  → [{now:.3f}] Candidati pubblicati: {', '.join(sorgenti)}")



        
    def command_loop(self):
        print("\nComandi disponibili:")
        print("  p → stampa albero semantico")
        print("  h → history posizioni di un oggetto")
        print("  t → path verso oggetti target")
        print("  g → target in linguaggio naturale (Gemini)")
        print("  x → deseleziona il target attivo")
        print("  v → verifica posizioni (distanze a coppie + z)")
        print("  q → esci\n")

        while rclpy.ok():
            try:
                cmd = input("Comando: ").strip().lower()
                if cmd == 'p':
                    with self.lock:
                        sg = list(scene_graph)
                    print_scene_graph(sg)
                elif cmd == 'h':
                    descrizione = input("Di quale oggetto vuoi la history? (puoi descriverlo a parole tue) ")
                    # Snapshot preso QUI, dopo l'input() bloccante e prima di
                    # leggere scene_graph, cosi' il lock non resta mai tenuto
                    # durante un'attesa da tastiera o una chiamata di rete a
                    # Gemini (vedi TODO piu' sopra sulla race con frame_callback).
                    with self.lock:
                        sg = list(scene_graph)
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    scene_json = scene_graph_to_json(sg, FREQ_THRESHOLD)
                    labels = resolve_targets(descrizione, scene_json)
                    label = labels[0] if labels else descrizione
                    history = get_position_history(sg, label)
                    if history:
                        print(f"History di '{label}': {history}")
                    else:
                        print(f"Oggetto '{label}' non trovato nella scena.")
                elif cmd == 't':
                    descrizione = input("Oggetti target (puoi descriverli a parole tue, separati da virgola): ")
                    with self.lock:
                        sg = list(scene_graph)
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    scene_json = scene_graph_to_json(sg, FREQ_THRESHOLD)
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
                            for node in sg:
                                if node['label'] == label and node['count'] >= FREQ_THRESHOLD:
                                    pos = node.get('position')
                                    if pos:
                                        pos_metri = pixel_to_meters_3d(pos[0], pos[1], current_z, CAMERA_MATRIX)
                                        path.append({'label': label, 'position_pixel': pos, 'position_metri': pos_metri})
                                    break
                    else:
                        path = get_path_to_targets_meters(sg, target_labels, SCALA_PIXEL_METRI)
                    if path:
                        print("\nPATH VERSO GLI OBIETTIVI:")
                        for step, target in enumerate(path):
                            print(f"  Step {step+1}: {target['label']} → pixel: {target['position_pixel']} | metri: {target['position_metri']}")
                    else:
                        print("Nessun oggetto trovato.")

                elif cmd == 'g':
                    from demo.gemini_retrieval import scene_graph_to_json, resolve_targets
                    descrizione = input("Descrivi il target a parole tue: ")
                    with self.lock:
                        sg = list(scene_graph)
                    scene_json = scene_graph_to_json(sg, FREQ_THRESHOLD)
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
                            for node in sg:
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

                elif cmd == 'x':
                    if self.arbiter is not None:
                        # command_loop gira su un thread separato dal ciclo SGG
                        # che fa avanzare l'arbitro: stesso lock del grafo.
                        with self.lock:
                            self.arbiter.reset()
                        print("  [arbitro ombra] scelta azzerata.")
                    if self.active_target_label is None:
                        print("Nessun target attivo da deselezionare.")
                    else:
                        print(f"  → Deselezionato target attivo: '{self.active_target_label}'")
                        self.active_target_label = None
                        
                elif cmd == 'v':
                    # Verifica posizioni: distanze a coppie + posizione relativa al marker ArUco
                    import itertools
                    with self.lock:
                        sg = list(scene_graph)
                    oggetti = []
                    for node in sg:
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
