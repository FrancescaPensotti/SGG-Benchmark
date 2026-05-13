import torch
import clip
import cv2
import numpy as np
from PIL import Image
import sys
import os
from query_scene import get_path_to_targets_meters, get_position_history, get_current_position
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.onnx_model import SGG_ONNX_Model
from demo.query_scene import get_position_history, get_current_position, get_path_to_targets
# ── Configurazione ──────────────────────────────────────────
ONNX_PATH = "checkpoints/VG150/react++_yolov8m/model.onnx"
SIMILARITY_THRESHOLD = 0.85  # soglia per considerare due oggetti "lo stesso"
FREQ_THRESHOLD = 10           # relazione deve essere vista almeno N volte
BLACKLIST_OBJECTS = {
    'table', 'floor', 'wall', 'ceiling', 'chair',
    'ground', 'background', 'window', 'door'
}
SCALA_PIXEL_METRI = 0.00105  # da calibrare in lab con oggetto di dimensione nota
DISAPPEAR_THRESHOLD = 30  # frame consecutivi senza vedere l'oggetto → rimosso - da valutare che valore dare

# ── Mappe relazioni VG150 → MomaGraph ───────────────────────
# Sempre attive, usate dal filtro relazioni
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

# ── Albero semantico incrementale ───────────────────────────
# Struttura: lista di nodi, ognuno con:
# - label: nome dell'oggetto
# - embedding: vettore CLIP 512D
# - relazioni: dizionario (pred, obj_node) -> count
# - count: quante volte lo abbiamo visto
# - position: centroide bounding box (cx, cy) in pixel
scene_graph = []

def get_clip_embedding(image, box):
    """Ritaglia l'oggetto dall'immagine e calcola il suo embedding CLIP."""
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
    """
    Cerca se un oggetto con questa label e embedding simile
    esiste gia nell'albero. Restituisce l'indice o -1.
    """
    for i, node in enumerate(scene_graph):
        if node['label'] == label:
            sim = torch.cosine_similarity(node['embedding'], embedding).item()
            if sim >= SIMILARITY_THRESHOLD:
                return i
    return -1

def update_scene_graph(image, bboxes, rels):
    """Aggiorna l'albero semantico con le nuove osservazioni."""
    box_to_node = {}

    # 1. Per ogni oggetto rilevato, aggiorna o crea un nodo
    for i, box in enumerate(bboxes):
        cls_idx = int(box[5])
        label = sgg.stats['obj_classes'].get(cls_idx, str(cls_idx))
        if label in BLACKLIST_OBJECTS:  # non considera gli oggetti nella blacklist
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
            box_to_node[i] = existing_idx
            scene_graph[existing_idx]['frames_not_seen'] = 0  # resetta contatore scomparsa
        else:
            new_node = {
                'label': label,
                'embedding': embedding,
                'relazioni': {},  # dizionario: (pred, obj_node) -> count
                'count': 1,
                'position': (cx, cy),
                'position_history': [(cx, cy)],
                'frames_not_seen': 0,   # per rimuovere oggetti scomparsi
            }
            scene_graph.append(new_node)
            box_to_node[i] = len(scene_graph) - 1

    # 2. Per ogni relazione, aggiorna gli archi nell'albero
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

    # Incrementa frames_not_seen per oggetti non visti in questo frame e rimuovi quelli spariti (tolti completamente dalla memoria dell'albero)
    seen_nodes = set(box_to_node.values())
    to_remove = []
    for i, node in enumerate(scene_graph):
        if i not in seen_nodes:
            node['frames_not_seen'] += 1
            if node['frames_not_seen'] >= DISAPPEAR_THRESHOLD:
               print(f"  ⚠️ '{node['label']}' sparito dalla scena — probabilmente preso dal robot.")
               to_remove.append(i)

    # Rimuovi in ordine inverso per non alterare gli indici
    for i in sorted(to_remove, reverse=True):
     scene_graph.pop(i)
# ══════════════════════════════════════════════════════════════
# VERSIONE 1: SGG + CLIP - INIZIO

def print_scene_graph():
    """Stampa lo stato attuale dell'albero semantico con filtro frequenza + mappa."""
    print("\n" + "="*50)
    print(f"ALBERO SEMANTICO — {len(scene_graph)} oggetti nella scena")
    print("="*50)
    for i, node in enumerate(scene_graph):
        pos = node.get('position', 'N/A')
        history = node.get('position_history', [])
        print(f"  [{i}] {node['label']} (visto {node['count']} volte) — pos attuale: {pos} — storia: {len(history)} punti")
        for (pred, obj_idx), count in node['relazioni'].items():
            # Filtro 1: frequenza minima
            if count < FREQ_THRESHOLD:
                continue
            # Filtro 2: deve avere significato funzionale o spaziale
            if pred not in VG_TO_FUNCTIONAL and pred not in VG_TO_SPATIAL:
                continue
            print(f"       --({pred})--> {scene_graph[obj_idx]['label']} [vista {count}x]")
    print("="*50 + "\n")
    

# VERSIONE 1: SGG + CLIP - FINE
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# ALTERNATIVA — Formato JSON stile MomaGraph
# ══════════════════════════════════════════════════════════════

# VERSIONE 2: MomaGraph-INIZIO
# ══════════════════════════════════════════════════════════════

# RELAZIONI FUNZIONALI disponibili in MomaGraph:
# "openorclose", "adjust", "control", "providepower",
# "activate", "pairwith"
#
# RELAZIONI SPAZIALI disponibili in MomaGraph:
# "left_of", "right_of", "in_front_of", "behind",
# "higher_than", "lower_than", "close", "far", "touching"
#
# FUNCTION_TYPES disponibili:
# "parameter_adjustment", "device_control", "open_close_control",
# "water_flow_control", "power_supply", "special_function", "assembly"
#
# ACTION_TYPES disponibili:
# "press", "rotate", "pull", "open", "push", "close", "insert"

# def print_scene_graph():
#     """Stampa l'albero in formato JSON stile MomaGraph."""
#     import json
#
#     nodes = [node['label'] for node in scene_graph]
#
#     edges = []
#     for i, node in enumerate(scene_graph):
#         for (pred, obj_idx), count in node['relazioni'].items():
#             # Filtro 1: frequenza minima
#             if count < FREQ_THRESHOLD:
#                 continue
#             # Filtro 2: deve avere significato funzionale o spaziale
#             if pred not in VG_TO_FUNCTIONAL and pred not in VG_TO_SPATIAL:
#                 continue
#
#             functional = VG_TO_FUNCTIONAL.get(pred, None)
#             spatial = VG_TO_SPATIAL.get(pred, None)
#
#             edge = {
#                 "object1": node['label'],
#                 "object2": scene_graph[obj_idx]['label'],
#             }
#
#             if functional:
#                 edge["functional_relationship"] = functional
#             if spatial:
#                 edge["spatial_relations"] = [spatial]
#
#             edge["is_touching"] = pred in ['holding', 'on', 'in',
#                                            'attached to', 'touching',
#                                            'covering', 'plugged into']
#             edges.append(edge)
#
#     unique_edges = []
#     seen = set()
#     for e in edges:
#         key = (e['object1'], e.get('functional_relationship',''),
#                e['object2'])
#         if key not in seen:
#             seen.add(key)
#             unique_edges.append(e)
#
#    # Posizioni attuali e storia di ogni oggetto
#     positions = {}
#     for node in scene_graph:
#         if node['count'] >= FREQ_THRESHOLD and node.get('position'):
#             positions[node['label']] = {
#                 "current": list(node['position']),
#                 "history": [list(p) for p in node.get('position_history', [])]
#             }
#
#     momagraph_json = {
#         "nodes": list(set(nodes)),
#         "edges": unique_edges,
#         "n_objects_seen": len(scene_graph),
#         "positions": positions,
#     }
#
#     print("\n" + "="*50)
#     print("SCENE GRAPH — Formato MomaGraph")
#     print("="*50)
#     print(json.dumps(momagraph_json, indent=2))
#     print("="*50 + "\n")

# VERSIONE 2: MomaGraph-FINE
# ══════════════════════════════════════════════════════════════


# ── Loop principale ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Webcam aperta. Premi 'q' per uscire, 'p' per stampare l'albero.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        result = sgg.predict(frame, visu_type='video')
        img, dbg = result
        
        if dbg is not None:
            bboxes, rels = dbg
            update_scene_graph(frame, bboxes, rels)

    frame_count += 1

    cv2.imshow("Incremental Scene Graph", img if 'img' in locals() else frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        print_scene_graph()
    
    elif key == ord('h'):
         label = input("Di quale oggetto vuoi la history? ")
         history = get_position_history(scene_graph, label)
         if history:
            print(f"History di '{label}': {history}")
         else:
            print(f"Oggetto '{label}' non trovato nel grafo.")

    elif key == ord('t'):
         labels = input("Inserisci gli oggetti in ordine separati da virgola: ")
         target_labels = [l.strip() for l in labels.split(',')]
         path = get_path_to_targets_meters(scene_graph, target_labels, SCALA_PIXEL_METRI)
         if path:
            print("\nPATH VERSO GLI OBIETTIVI:")
         for step, target in enumerate(path):
            print(f"  Step {step+1}: {target['label']} → pixel: {target['position_pixel']} | metri: {target['position_metri']}")
         else:
            print("Nessun oggetto trovato nel grafo.")

cap.release()
cv2.destroyAllWindows()

# Stampa il risultato finale
print_scene_graph()
