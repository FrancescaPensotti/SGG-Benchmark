#!/usr/bin/env python3
"""
Test standalone della logica di avanzamento verso l'oggetto connesso (Fase C),
SENZA bisogno di ROS/camera/SGG. Riproduce advance_to_next_object() usando un
scene_graph finto, per verificare la logica di selezione prima di testarla
nella catena completa.

TODO (audit): advance_to_next_object_test() qui sotto e' una copia a mano di
SGGNode.advance_to_next_object() in sgg_ros_node.py, non e' importata da li' —
una modifica futura al metodo reale non viene rilevata da questo test finche'
qualcuno non ricorda di rispecchiarla qui.
"""

# Sottoinsieme del dizionario reale, sufficiente per i test.
# Le chiavi sono i predicati grezzi VG150 così come escono da SGG-Benchmark.
VG_TO_FUNCTIONAL = {
    'holding': 'control',
    'attached to': 'pairwith',
    'connected to': 'pairwith',
    'plugged into': 'providepower',
}


def make_node(label, relazioni):
    return {'label': label, 'relazioni': relazioni}


def advance_to_next_object_test(scene_graph, grasped_label):
    """Stessa logica di advance_to_next_object(), ma riceve scene_graph come
    parametro (invece di leggerlo come globale) e restituisce il risultato
    invece di assegnarlo a self.active_target_label."""
    grasped_node = next((n for n in scene_graph if n['label'] == grasped_label), None)

    if grasped_node is None:
        print(f"  ⚠️ '{grasped_label}' non più nel grafo, nessun proseguimento automatico.")
        return None

    best_rel, best_count = None, 0
    for (pred, obj_idx), count in grasped_node['relazioni'].items():
        if pred in VG_TO_FUNCTIONAL and count > best_count:
            best_rel, best_count = (pred, obj_idx), count

    if best_rel is None:
        print(f"  ℹ️ Nessuna relazione funzionale per '{grasped_label}': nessun successivo.")
        return None

    pred, obj_idx = best_rel
    next_label = scene_graph[obj_idx]['label']
    print(f"  → Grasp di '{grasped_label}' rilevato. Prossimo target: '{next_label}' ({pred}, {best_count}x)")
    return next_label


# ── Test 1: funzionale batte spaziale, anche con conteggio più basso ────────
print("Test 1: 'bottle' ha una relazione funzionale (conteggio basso) e una")
print("spaziale (conteggio alto) — deve vincere quella funzionale.")
scene_graph_1 = [
    make_node('bottle', {('attached to', 1): 6, ('close', 2): 15}),
    make_node('cap', {}),
    make_node('glass', {}),
]
result_1 = advance_to_next_object_test(scene_graph_1, 'bottle')
assert result_1 == 'cap', f"Atteso 'cap', ottenuto '{result_1}'"
print("  ✅ Passato\n")

# ── Test 2: nessuna relazione funzionale disponibile ─────────────────────────
print("Test 2: 'table' ha solo relazioni spaziali — nessun successivo atteso.")
scene_graph_2 = [
    make_node('table', {('above', 1): 10, ('close', 2): 3}),
    make_node('floor', {}),
    make_node('wall', {}),
]
result_2 = advance_to_next_object_test(scene_graph_2, 'table')
assert result_2 is None, f"Atteso None, ottenuto '{result_2}'"
print("  ✅ Passato\n")

# ── Test 3: l'oggetto graspato non è più nel grafo (es. rimosso dal decay) ──
print("Test 3: si prova ad avanzare da 'bottle', ma non è più nel grafo.")
scene_graph_3 = [make_node('cup', {})]
result_3 = advance_to_next_object_test(scene_graph_3, 'bottle')
assert result_3 is None, f"Atteso None, ottenuto '{result_3}'"
print("  ✅ Passato\n")

# ── Test 4: due relazioni funzionali concorrenti, vince il conteggio più alto ─
print("Test 4: 'screwdriver' ha due relazioni funzionali — vince quella col")
print("conteggio più alto ('connected to' -> screw, 9 > 'holding' -> drill, 5).")
scene_graph_4 = [
    make_node('screwdriver', {('connected to', 1): 9, ('holding', 2): 5}),
    make_node('screw', {}),      # indice 1, collegato con 'connected to' (9x)
    make_node('drill', {}),      # indice 2, collegato con 'holding' (5x)
]
result_4 = advance_to_next_object_test(scene_graph_4, 'screwdriver')
assert result_4 == 'screw', f"Atteso 'screw', ottenuto '{result_4}'"
print("  ✅ Passato\n")

print("Tutti i test superati.")