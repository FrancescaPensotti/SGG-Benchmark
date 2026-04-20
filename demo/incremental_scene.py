import torch
import clip
import cv2
import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.onnx_model import SGG_ONNX_Model

# ── Configurazione ──────────────────────────────────────────
ONNX_PATH = "checkpoints/VG150/react++_yolov8m/model.onnx"
SIMILARITY_THRESHOLD = 0.85  # soglia per considerare due oggetti "lo stesso"

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
# - relazioni: lista di (relazione, idx_altro_nodo)
# - count: quante volte lo abbiamo visto
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
    # Mappa: idx bounding box -> idx nodo nell'albero
    box_to_node = {}

    # 1. Per ogni oggetto rilevato, aggiorna o crea un nodo
    for i, box in enumerate(bboxes):
        cls_idx = int(box[5])
        label = sgg.stats['obj_classes'].get(cls_idx, str(cls_idx))
        
        embedding = get_clip_embedding(image, box)
        if embedding is None:
            continue

        existing_idx = find_existing_node(label, embedding)
        
        if existing_idx >= 0:
            # Oggetto gia visto — aggiorna il nodo esistente
            scene_graph[existing_idx]['count'] += 1
            scene_graph[existing_idx]['embedding'] = embedding  # aggiorna embedding
            box_to_node[i] = existing_idx
        else:
            # Oggetto nuovo — crea un nodo nuovo
            new_node = {
                'label': label,
                'embedding': embedding,
                'relazioni': [],
                'count': 1
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
                
                # Aggiungi relazione se non esiste gia
                rel_tuple = (pred, obj_node)
                if rel_tuple not in scene_graph[subj_node]['relazioni']:
                    scene_graph[subj_node]['relazioni'].append(rel_tuple)

# TOLGO COOMENTO PER SGG+CLIP 

# def print_scene_graph():
#     """Stampa lo stato attuale dell'albero semantico."""
#     print("\n" + "="*50)
#     print(f"ALBERO SEMANTICO — {len(scene_graph)} oggetti nella scena")
#     print("="*50)
#     for i, node in enumerate(scene_graph):
#         print(f"  [{i}] {node['label']} (visto {node['count']} volte)")
#         for pred, obj_idx in node['relazioni']:
#             print(f"       --({pred})--> {scene_graph[obj_idx]['label']}")
#     print("="*50 + "\n")

# 

# ══════════════════════════════════════════════════════════════
# ALTERNATIVA — Formato JSON stile MomaGraph
# Commenta/decommenta per passare da un formato all'altro
# ══════════════════════════════════════════════════════════════

# VERSIONE ATTIVA: formato testuale SGG-Benchmark
# def print_scene_graph():
#     """Stampa l'albero in formato testuale semplice."""
#     print("\n" + "="*50)
#     print(f"ALBERO SEMANTICO — {len(scene_graph)} oggetti nella scena")
#     print("="*50)
#     for i, node in enumerate(scene_graph):
#         print(f"  [{i}] {node['label']} (visto {node['count']} volte)")
#         for pred, obj_idx in node['relazioni']:
#             print(f"       --({pred})--> {scene_graph[obj_idx]['label']}")
#     print("="*50 + "\n")

# ──────────────────────────────────────────────────────────────
# VERSIONE COMMENTATA: formato JSON stile MomaGraph
# Per attivarla: commenta la funzione sopra e decommenta questa
# ──────────────────────────────────────────────────────────────

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

# MAPPA relazioni VG150 → relazioni funzionali MomaGraph
# Usata per convertire le relazioni di SGG-Benchmark nel formato MomaGraph
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

# MAPPA relazioni VG150 → relazioni spaziali MomaGraph
VG_TO_SPATIAL = {
    'above':    'higher_than',
    'below':    'lower_than',
    'behind':   'behind',
    'in front of': 'in_front_of',
    'left of':  'left_of',
    'right of': 'right_of',
    'near':     'close',
    'next to':  'close',
    'on':       'touching',
    'in':       'touching',
    'touching': 'touching',
}

# COMMENTO QUI PER PASSARE A SGG+CLIP
def print_scene_graph():
    """Stampa l'albero in formato JSON stile MomaGraph."""
    import json

    nodes = [node['label'] for node in scene_graph]

    edges = []
    for i, node in enumerate(scene_graph):
        for pred, obj_idx in node['relazioni']:

            # Converti la relazione VG150 in funzionale o spaziale
            functional = VG_TO_FUNCTIONAL.get(pred, None)
            spatial = VG_TO_SPATIAL.get(pred, None)

            edge = {
                "object1": node['label'],
                "object2": scene_graph[obj_idx]['label'],
            }

            if functional:
                edge["functional_relationship"] = functional
            if spatial:
                edge["spatial_relations"] = [spatial]

            # is_touching: True se la relazione implica contatto fisico
            edge["is_touching"] = pred in ['holding', 'on', 'in',
                                           'attached to', 'touching',
                                           'covering', 'plugged into']

            edges.append(edge)

    # Rimuovi duplicati negli archi
    unique_edges = []
    seen = set()
    for e in edges:
        key = (e['object1'], e.get('functional_relationship',''),
               e['object2'])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    # Costruisci il JSON finale stile MomaGraph
    momagraph_json = {
        "nodes": list(set(nodes)),  # rimuovi nodi duplicati
        "edges": unique_edges,
        "n_objects_seen": len(scene_graph),
    }

    print("\n" + "="*50)
    print("SCENE GRAPH — Formato MomaGraph")
    print("="*50)
    print(json.dumps(momagraph_json, indent=2))
    print("="*50 + "\n")

# 

# ── Loop principale ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Webcam aperta. Premi 'q' per uscire, 'p' per stampare l'albero.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Elabora ogni 5 frame per non sovraccaricare la CPU
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

cap.release()
cv2.destroyAllWindows()

# Stampa il risultato finale
print_scene_graph()
