#!/usr/bin/env python3
"""
Test standalone del meccanismo di forgetting factor (confidence decay) e della
logica di target attivo, SENZA bisogno di ROS/camera. Simula manualmente
l'aggiornamento del scene_graph nel tempo.
"""

CONFIDENCE_DECAY = 0.98
CONFIDENCE_REMOVE_THRESHOLD = 0.05

def make_node(label):
    return {
        'label': label,
        'count': 1,
        'position': (100, 100),
        'frames_not_seen': 0,
        'confidence': 1.0,
    }

def decay_step(scene_graph, seen_labels):
    """Simula un ciclo: gli oggetti in seen_labels vengono 'rivisti' (reset),
    gli altri decadono."""
    to_remove = []
    for i, node in enumerate(scene_graph):
        if node['label'] in seen_labels:
            node['frames_not_seen'] = 0
            node['confidence'] = 1.0
        else:
            node['frames_not_seen'] += 1
            node['confidence'] *= CONFIDENCE_DECAY
            if node['confidence'] < CONFIDENCE_REMOVE_THRESHOLD:
                to_remove.append(i)
    for i in sorted(to_remove, reverse=True):
        print(f"  ⚠️ '{scene_graph[i]['label']}' rimosso al ciclo (confidence sotto soglia)")
        scene_graph.pop(i)

def republish_active_target(scene_graph, active_label):
    """Stessa logica di republish_active_target(), ma stampa invece di pubblicare
    su ROS, per verificarne il comportamento in isolamento."""
    if active_label is None:
        print("  (nessun target attivo)")
        return
    for node in scene_graph:
        if node['label'] == active_label:
            if node['confidence'] >= CONFIDENCE_REMOVE_THRESHOLD:
                print(f"  → ripubblicherei '{active_label}' (confidence={node['confidence']:.3f})")
            return
    print(f"  → '{active_label}' non trovato nel grafo: nessuna pubblicazione")


# ── Scenario di test ──────────────────────────────────────────
scene_graph = [make_node("bottle"), make_node("cup")]
active_target_label = "bottle"

print("Ciclo 0 (appena rilevati, target 'bottle' selezionato con 't'):")
republish_active_target(scene_graph, active_target_label)

print("\nSimulo 'bottle' che esce dal field of view per 200 cicli consecutivi:")
for cycle in range(1, 201):
    decay_step(scene_graph, seen_labels={"cup"})  # solo 'cup' resta visibile
    if cycle in (1, 10, 50, 100, 140, 148, 150, 160, 200):
        conf = next((n['confidence'] for n in scene_graph if n['label'] == "bottle"), None)
        print(f"  Ciclo {cycle}: confidence 'bottle' = {conf}")
        republish_active_target(scene_graph, active_target_label)

print("\nStato finale del grafo:", [n['label'] for n in scene_graph])

#secondo test: 'bottle' sparisce per 100 cicli, poi ricompare
print("\n\nTest 2: 'bottle' sparisce per 100 cicli, poi ricompare:")
scene_graph2 = [make_node("bottle")]
for cycle in range(1, 101):
    decay_step(scene_graph2, seen_labels=set())  # bottle non vista
conf_before = scene_graph2[0]['confidence']
print(f"  Dopo 100 cicli sparita: confidence = {conf_before:.3f}")

decay_step(scene_graph2, seen_labels={"bottle"})  # ricompare
conf_after = scene_graph2[0]['confidence']
print(f"  Ricompare -> confidence reset a: {conf_after:.3f}")