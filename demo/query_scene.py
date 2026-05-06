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

def get_path_to_targets(scene_graph, target_labels, freq_threshold=10):
    """
    Dato un elenco ordinato di oggetti target, restituisce
    la sequenza di posizioni (cx, cy) da raggiungere in ordine.
    
    Esempio:
        get_path_to_targets(scene_graph, ["bottle", "phone"])
        → [(82, 347), (261, 321)]
    """
    path = []
    for label in target_labels:
        for node in scene_graph:
            if node['label'] == label and node['count'] >= freq_threshold:
                pos = node.get('position')
                if pos:
                    path.append({
                        'label': label,
                        'position': pos
                    })
                break  # prendi il primo nodo che matcha
    return path

def pixel_to_meters_2d(cx_pixel, cy_pixel, scala):
    """
    Converte coordinate pixel in metri sul piano 2D.
    
    Args:
        cx_pixel, cy_pixel: centroide oggetto in pixel
        scala: metri per pixel (da calibrare in lab)
    
    Returns:
        (x_metri, y_metri): coordinate reali nel piano 2D
    """
    x_metri = cx_pixel * scala
    y_metri = cy_pixel * scala
    return (x_metri, y_metri)


def get_path_to_targets_meters(scene_graph, target_labels, scala, freq_threshold=10):
    """
    Restituisce il path verso gli oggetti target in metri (piano 2D).
    """
    path = []
    for label in target_labels:
        for node in scene_graph:
            if node['label'] == label and node['count'] >= freq_threshold:
                pos = node.get('position')
                if pos:
                    pos_metri = pixel_to_meters_2d(pos[0], pos[1], scala)
                    path.append({
                        'label': label,
                        'position_pixel': pos,
                        'position_metri': pos_metri
                    })
                break
    return path
