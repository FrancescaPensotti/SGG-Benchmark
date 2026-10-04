"""
Salvataggio e disegno delle richieste a GraspNet (04/10/2026), per la figura
del Cap. 5 con le prese candidate e la presa scelta.

graspnet_node.py chiama save_request() dopo aver pubblicato l'orientamento:
il file non cambia niente nella catena che muove il robot. Ogni richiesta
diventa un .npz con l'immagine a colori, la profondita' (mm), gli intrinseci,
il riquadro del target con margine, i punti target di SGG, tutte le prese
restituite dal server (in frame camera) e l'indice di quella scelta.

Modulo puro, senza ROS: lo usano anche lo script di disegno e il test.

Uso da riga di comando (disegna una richiesta salvata):
    python3 demo/graspnet_records.py $LAB/graspnet/grasp_<ora>.npz [uscita.png]
"""
import datetime
import os
import sys

import numpy as np

# Geometria della pinza per il disegno, come nel visualizzatore di
# graspnetAPI (plot_gripper_pro_max): le dita vanno da -DEPTH_BASE a +DEPTH
# lungo l'asse di avvicinamento x del frame di presa, alle quote -w/2 e +w/2
# lungo y. Il server non restituisce la profondita' di inserimento, quindi si
# usa un valore fisso: serve solo al disegno.
DEPTH_BASE = 0.02
DEPTH = 0.02
TAIL = 0.04


def default_save_dir():
    """Stessa cartella dei log dei profili di laboratorio: $LAB se impostata,
    altrimenti ~/lab_logs/<data di oggi>; i file vanno nella sottocartella
    graspnet."""
    base = os.environ.get('LAB') or os.path.join(
        os.path.expanduser('~'), 'lab_logs', datetime.date.today().isoformat())
    return os.path.join(base, 'graspnet')


def save_request(save_dir, color_bgr, depth_mm, intrinsics, bbox, target_points,
                 result, chosen, params):
    """Salva una richiesta in save_dir/grasp_<ora>.npz e restituisce il
    percorso. chosen e' la presa scelta da select_grasp (None se nessuna):
    il suo indice si ritrova confrontando la posizione con quelle del server,
    e la sua rotazione e' quella del gemello effettivamente usato."""
    os.makedirs(save_dir, exist_ok=True)
    grasps = (result or {}).get('grasps') or []
    n = len(grasps)
    translations = np.array([g['translation'] for g in grasps], dtype=float).reshape(n, 3)
    rotations = np.array([g['rotation_matrix'] for g in grasps], dtype=float).reshape(n, 3, 3)
    widths = np.array([g.get('width', np.nan) for g in grasps], dtype=float)
    scores = np.array([g.get('score', np.nan) for g in grasps], dtype=float)

    chosen_index = -1
    chosen_rotation = np.full((3, 3), np.nan)
    if chosen is not None:
        t = np.array(chosen['translation'], dtype=float)
        if n:
            d = np.linalg.norm(translations - t, axis=1)
            if d.min() < 1e-9:
                chosen_index = int(d.argmin())
        chosen_rotation = np.array(chosen['rotation_matrix'], dtype=float)

    stamp = datetime.datetime.now()
    path = os.path.join(save_dir, 'grasp_' + stamp.strftime('%H%M%S_%f')[:-3] + '.npz')
    np.savez_compressed(
        path,
        color_bgr=color_bgr.astype(np.uint8),
        depth_mm=depth_mm.astype(np.uint16),
        intrinsics=np.array(intrinsics, dtype=float),          # fx, fy, cx, cy
        bbox=np.array(bbox if bbox is not None else [], dtype=float),
        target_points=np.array(target_points, dtype=float).reshape(-1, 3),
        success=bool((result or {}).get('success', False)),
        reason=str((result or {}).get('reason', '')),
        translations=translations, rotations=rotations, widths=widths, scores=scores,
        chosen_index=chosen_index, chosen_rotation=chosen_rotation,
        max_grasp_width=float(params.get('max_grasp_width', np.nan)),
        min_grasp_score=float(params.get('min_grasp_score', np.nan)),
        target_radius=float(params.get('target_radius', np.nan)),
        time=stamp.isoformat(),
    )
    return path


def gripper_segments(translation, rotation, width):
    """Segmenti 3D (frame camera) che disegnano la pinza: due dita, la base e
    la coda lungo l'asse di avvicinamento."""
    c = np.asarray(translation, dtype=float)
    R = np.asarray(rotation, dtype=float)
    p = lambda x, y: c + R @ np.array([x, y, 0.0])
    w = width / 2.0
    return [
        (p(-DEPTH_BASE, -w), p(DEPTH, -w)),        # dito sinistro
        (p(-DEPTH_BASE, w), p(DEPTH, w)),          # dito destro
        (p(-DEPTH_BASE, -w), p(-DEPTH_BASE, w)),   # base
        (p(-DEPTH_BASE, 0.0), p(-DEPTH_BASE - TAIL, 0.0)),  # coda
    ]


def project(points, intrinsics):
    fx, fy, cx, cy = intrinsics
    pts = np.atleast_2d(points)
    return np.stack([fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy], axis=1)


def draw_request(path, out_path=None):
    """Disegna sull'immagine a colori il riquadro con margine, il punto
    target di SGG, le prese candidate (grigio: piu' larghe della pinza o
    lontane dal target; arancione: valide) e la presa scelta (verde)."""
    import cv2
    data = np.load(path)
    img = data['color_bgr'].copy()
    K = data['intrinsics']
    bbox = data['bbox']
    if bbox.size == 4:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 0), 2)
    targets = data['target_points']
    for t in targets:
        u, v = project(t, K)[0].astype(int)
        cv2.drawMarker(img, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)

    order = list(range(len(data['scores'])))
    chosen = int(data['chosen_index'])
    for i in order:
        if i == chosen:
            continue
        width = data['widths'][i]
        near = (len(targets) == 0 or
                np.min(np.linalg.norm(targets - data['translations'][i], axis=1)) < data['target_radius'])
        valid = width <= data['max_grasp_width'] and near
        color = (0, 140, 255) if valid else (160, 160, 160)
        for a, b in gripper_segments(data['translations'][i], data['rotations'][i], width):
            (u1, v1), (u2, v2) = project(np.stack([a, b]), K).astype(int)
            cv2.line(img, (u1, v1), (u2, v2), color, 1, cv2.LINE_AA)
    if chosen >= 0:
        rot = data['chosen_rotation'] if np.all(np.isfinite(data['chosen_rotation'])) else data['rotations'][chosen]
        for a, b in gripper_segments(data['translations'][chosen], rot, data['widths'][chosen]):
            (u1, v1), (u2, v2) = project(np.stack([a, b]), K).astype(int)
            cv2.line(img, (u1, v1), (u2, v2), (0, 200, 0), 3, cv2.LINE_AA)

    out_path = out_path or os.path.splitext(path)[0] + '.png'
    cv2.imwrite(out_path, img)
    return out_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(draw_request(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
