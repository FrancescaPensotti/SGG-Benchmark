# demo/query_scene.py
# Funzioni per interrogare il grafo semantico e recuperare posizioni

def get_position_history(scene_graph, label):
    """Restituisce la storia completa delle posizioni di un oggetto."""
    for node in scene_graph:
        if node['label'] == label:
            return node.get('position_history', [])
    return []

def get_current_position(scene_graph, label):
    """Restituisce la posizione attuale di un oggetto."""
    for node in scene_graph:
        if node['label'] == label:
            return node.get('position', None)
    return None

def get_last_n_positions(scene_graph, label, n=5):
    """Restituisce le ultime N posizioni di un oggetto."""
    history = get_position_history(scene_graph, label)
    return history[-n:]
