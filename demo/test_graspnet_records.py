"""
Test di demo/graspnet_records.py (04/10/2026): salvataggio di una richiesta
finta, rilettura dei campi e disegno. Nessuna rete, nessun ROS.

    python demo/test_graspnet_records.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from demo.graspnet_records import save_request, draw_request, project, gripper_segments

ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
    else:
        fail += 1
        print('FALLITO:', name)


K = (912.1, 911.9, 650.9, 383.9)
color = np.full((720, 1280, 3), 90, dtype=np.uint8)
depth = np.full((720, 1280), 400, dtype=np.uint16)
target = np.array([0.02, 0.01, 0.40])
R_down = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # avvicinamento lungo z camera
grasps = [
    {'score': 0.8, 'width': 0.05, 'rotation_matrix': R_down.tolist(), 'translation': (target + [0.0, 0.0, 0.02]).tolist()},
    {'score': 0.6, 'width': 0.10, 'rotation_matrix': R_down.tolist(), 'translation': (target + [0.03, 0.0, 0.02]).tolist()},
    {'score': 0.5, 'width': 0.04, 'rotation_matrix': R_down.tolist(), 'translation': (target + [0.30, 0.0, 0.0]).tolist()},
]
result = {'success': True, 'grasps': grasps}
flipped = R_down @ np.diag([1.0, -1.0, -1.0])  # gemello a 180 gradi attorno a x
chosen = dict(grasps[0], rotation_matrix=flipped)
params = {'max_grasp_width': 0.075, 'min_grasp_score': 0.4, 'target_radius': 0.15}

with tempfile.TemporaryDirectory() as d:
    path = save_request(d, color, depth, K, [500, 300, 800, 480], [target], result, chosen, params)
    check('file creato', os.path.isfile(path))
    data = np.load(path)
    check('immagine a colori', data['color_bgr'].shape == (720, 1280, 3) and data['color_bgr'].dtype == np.uint8)
    check('profondita in mm', data['depth_mm'].dtype == np.uint16 and int(data['depth_mm'][0, 0]) == 400)
    check('intrinseci', np.allclose(data['intrinsics'], K))
    check('riquadro', data['bbox'].tolist() == [500, 300, 800, 480])
    check('target', np.allclose(data['target_points'], [target]))
    check('tre prese', data['translations'].shape == (3, 3) and data['rotations'].shape == (3, 3, 3))
    check('larghezze e punteggi', np.allclose(data['widths'], [0.05, 0.10, 0.04]) and np.allclose(data['scores'], [0.8, 0.6, 0.5]))
    check('indice della scelta', int(data['chosen_index']) == 0)
    check('rotazione del gemello usato', np.allclose(data['chosen_rotation'], flipped))
    check('esito', bool(data['success']) and str(data['reason']) == '')
    png = draw_request(path)
    check('disegno salvato', os.path.isfile(png))
    import cv2
    img = cv2.imread(png)
    u, v = project(target, K)[0].astype(int)
    check('pixel verdi della presa scelta', np.any(np.all(img == (0, 200, 0), axis=2)))
    check('croce rossa sul target', tuple(img[v, u]) == (0, 0, 255))

    # Richiesta senza prese (server: no_valid_grasps) e richiesta senza scelta.
    p2 = save_request(d, color, depth, K, None, [], {'success': False, 'reason': 'no_valid_grasps'}, None, params)
    d2 = np.load(p2)
    check('senza prese: nessuna presa', d2['translations'].shape == (0, 3) and int(d2['chosen_index']) == -1)
    check('senza prese: motivo', str(d2['reason']) == 'no_valid_grasps' and not bool(d2['success']))
    check('senza prese: riquadro vuoto', d2['bbox'].size == 0)
    check('senza prese: disegno', os.path.isfile(draw_request(p2)))
    p3 = save_request(d, color, depth, K, None, [target], result, None, params)
    check('senza scelta: indice -1', int(np.load(p3)['chosen_index']) == -1)

# Geometria della pinza: dita a distanza pari all'apertura.
segs = gripper_segments([0, 0, 0.4], np.eye(3), 0.06)
check('dita distanti quanto l\'apertura', np.isclose(np.linalg.norm(segs[0][0] - segs[1][0]), 0.06))

print(f'{ok}/{ok + fail} test superati')
sys.exit(1 if fail else 0)
