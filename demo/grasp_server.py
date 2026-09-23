import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify

# Percorso assoluto della repo graspnet-baseline (repo ufficiale, clonato
# in fase di setup GPU sulla VM). Serve per raggiungere i moduli models/,
# dataset/, utils/ che non sono installati come pacchetti pip, ma vanno
# aggiunti manualmente al sys.path per essere importabili.

ROOT_DIR = os.path.expanduser("~/graspnet-env/graspnet-baseline")
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

# Import dalla repo graspnet-baseline (via sys.path sopra) e da graspnetAPI
# (installata via pip nel venv). Nessun conflitto di nomi con ROS2 qui,
# perche' questo file NON importa rclpy/sensor_msgs — gira come processo
# separato, senza ROS2 (vedi nota architetturale sotto).

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from graspnetAPI import GraspGroup

# --- Configurazione inferenza ---
# Equivalenti ai parametri da riga di comando (argparse) di demo.py,
# qui fissi perche' questo server non viene lanciato con argomenti.

CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "checkpoint-rs.tar")
NUM_POINT = 20000       # punti campionati dalla point cloud per l'inferenza
NUM_VIEW = 300          # numero di viste candidate valutate da GraspNet
COLLISION_THRESH = 0.01 # soglia di collisione per il filtro anti-collisione
VOXEL_SIZE = 0.01       # dimensione voxel per il collision detector
TOP_K = 20              # grasp restituiti al client, ordinati per score

app = Flask(__name__)

# --- Caricamento modello ---
# ESEGUITO UNA SOLA VOLTA all'avvio del server (livello di modulo, non
# dentro una funzione chiamata ad ogni richiesta) — ricaricare il
# checkpoint ad ogni richiesta HTTP sarebbe troppo lento.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = GraspNet(
    input_feature_dim=0,
    num_view=NUM_VIEW,
    num_angle=12,
    num_depth=4,
    cylinder_radius=0.05,
    hmin=-0.02,
    hmax_list=[0.01, 0.02, 0.03, 0.04],
    is_training=False,
)
net.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
net.load_state_dict(checkpoint["model_state_dict"])
net.eval()

print(f"[grasp_server] Checkpoint caricato: {CHECKPOINT_PATH}")
print(f"[grasp_server] Device: {device}")

import base64

def decode_array(encoded_str, shape, dtype):
    """Ricostruisce un array numpy a partire dalla stringa base64 ricevuta
    nella richiesta HTTP. Il client (graspnet_node.py sul laptop/robot)
    codifica color/depth con base64.b64encode(array.tobytes()) prima di
    inviarli — qui si fa l'operazione inversa."""
    raw_bytes = base64.b64decode(encoded_str)
    array = np.frombuffer(raw_bytes, dtype=dtype)
    return array.reshape(shape)


def build_point_cloud(color, depth, fx, fy, cx, cy, depth_scale,
                       depth_min_mm, depth_max_mm, bbox=None):
    """Equivalente a get_and_process_data() di demo.py, ma con color/depth
    ricevuti dalla richiesta invece che catturati da una RealSense locale
    (qui non c'e' nessuna camera collegata, la VM riceve solo i dati gia'
    catturati altrove).

    depth_min_mm / depth_max_mm: soglie di profondita' in millimetri per
    isolare la zona di lavoro (workspace mask). NOTA: nel demo.py originale
    questi valori erano tarati per la scena dimostrativa del repo (oggetti
    a media distanza). Nel nostro caso GraspNet interviene quando il
    braccio e' gia' vicino all'oggetto (fase di grasp) — questi valori
    vanno quindi tarati empiricamente in lab sul nostro banco, non presi
    dal demo originale. Per questo sono parametri, non costanti fisse.

    bbox: [x1, y1, x2, y2] in pixel, opzionale (22/09/2026). Aggiunge un
    secondo filtro alla workspace mask, questa volta su X/Y invece che su
    Z: limita la point cloud alla zona intorno al target rilevato da SGG
    (bbox dell'oggetto + margine, calcolato lato client in graspnet_node.py
    a partire da bbox_margin_m). Non tocca la soglia di profondita'
    assoluta: il tavolo sotto l'oggetto resta nella point cloud (e quindi
    visibile al collision detector) perche' e' comunque dentro
    depth_min_mm/depth_max_mm — qui si toglie solo cio' che e' lontano
    lateralmente (sfondo, altri oggetti), non cio' che sta sotto.
    Se None, comportamento invariato (solo filtro di profondita').
    """
    # CameraInfo e' la struct attesa da create_point_cloud_from_depth_image
    # (definita in graspnet-baseline/utils/data_utils.py) — non e' la stessa
    # cosa di sensor_msgs.msg.CameraInfo usata nel nodo ROS2, e' solo un
    # contenitore semplice per gli intrinseci.
    camera = CameraInfo(
        width=depth.shape[1],
        height=depth.shape[0],
        fx=fx, fy=fy, cx=cx, cy=cy,
        scale=1.0 / depth_scale,
    )

    # Filtra profondita' non plausibili (rumore sensore, sfondo troppo
    # lontano) prima di generare la point cloud.
    depth_filtered = depth.copy()
    depth_filtered[depth_filtered < depth_min_mm] = 0
    depth_filtered[depth_filtered > depth_max_mm] = 0

    cloud = create_point_cloud_from_depth_image(depth_filtered, camera, organized=True)

    # Maschera di lavoro: esclude i punti a profondita' zero (invalidati
    # sopra) e limita la point cloud alla zona rilevante.
    workspace_mask = (depth_filtered > depth_min_mm) & (depth_filtered < depth_max_mm)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(round(x1)))
        y1 = max(0, int(round(y1)))
        x2 = min(depth.shape[1], int(round(x2)))
        y2 = min(depth.shape[0], int(round(y2)))
        xy_mask = np.zeros_like(workspace_mask)
        xy_mask[y1:y2, x1:x2] = True
        workspace_mask = workspace_mask & xy_mask

    cloud_masked = cloud[workspace_mask]
    color_masked = color[workspace_mask]

    # GraspNet richiede un numero fisso di punti in input (NUM_POINT).
    # Se ne abbiamo di piu', sotto-campioniamo; se ne abbiamo di meno,
    # ripetiamo alcuni punti a caso per raggiungere il numero richiesto.
    #
    # RNG con seed fisso (22/09/2026), non np.random globale: a parita' di
    # scena (stesso color+depth+bbox in ingresso) il sottocampionamento era
    # diverso a ogni chiamata, e per oggetti piccoli con pochi punti validi
    # questo da solo poteva cambiare quali prese venivano proposte -- uno dei
    # sospetti aperti sulla scarsa ripetibilita' vista in laboratorio. Con
    # seed fisso, la stessa identica richiesta rimandata al server produce
    # sempre lo stesso risultato: permette di distinguere "il sottocampionamento
    # e' la causa" (rimandando lo stesso payload si ottiene sempre la stessa
    # risposta) da "la scena reale cambia da una prova all'altra" (motivo
    # diverso, non risolto da questo fix).
    _rng = np.random.default_rng(42)
    if len(cloud_masked) >= NUM_POINT:
        idxs = _rng.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = _rng.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # Tensore pronto per la rete: batch di dimensione 1 (np.newaxis aggiunge
    # la dimensione batch attesa dal modello).
    cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

    end_points = {
        "point_clouds": cloud_tensor,
        "cloud_colors": color_sampled,
    }

    # cloud_masked (non sotto-campionata) serve dopo per il collision
    # detector, che lavora meglio con piu' punti possibili della scena.
    return end_points, cloud_masked
def run_inference(end_points):
    """Esegue l'inferenza GraspNet vera e propria sulla point cloud
    preparata da build_point_cloud(). Identico a get_grasps() di demo.py.
    """
    # Sposta i tensori sul device giusto (GPU se disponibile, vedi 'device'
    # definito a livello di modulo insieme al caricamento del modello).
    for key in end_points:
        if isinstance(end_points[key], torch.Tensor):
            end_points[key] = end_points[key].to(device)

    # torch.no_grad(): siamo in inferenza, non serve calcolare i gradienti
    # (risparmia memoria e tempo).
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)

    # grasp_preds[0] perche' abbiamo un batch di dimensione 1 (una sola
    # scena alla volta, vedi np.newaxis in build_point_cloud).
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)

    return gg


def filter_collisions(gg, cloud_points):
    """Rimuove i grasp che entrerebbero in collisione con la scena.
    Identico a collision_detection() di demo.py. cloud_points e' la point
    cloud NON sotto-campionata (piu' densa = rilevamento piu' affidabile).
    """
    detector = ModelFreeCollisionDetector(
        np.asarray(cloud_points),
        voxel_size=VOXEL_SIZE,
    )
    collision_mask = detector.detect(
        gg,
        approach_dist=0.05,
        collision_thresh=COLLISION_THRESH,
    )
    return gg[~collision_mask]

@app.route("/predict_grasp", methods=["POST"])
def predict_grasp():
    """Endpoint principale: riceve color+depth+intrinseci dal client
    (graspnet_node.py sul laptop/robot), esegue l'intera pipeline GraspNet,
    e ritorna il grasp migliore come JSON.

    Formato atteso nel body JSON della richiesta:
    {
        "color": "<base64>", "color_shape": [H, W, 3], "color_dtype": "uint8",
        "depth": "<base64>", "depth_shape": [H, W],    "depth_dtype": "float32",
        "fx": ..., "fy": ..., "cx": ..., "cy": ...,
        "depth_scale": ...,          # metri per unita' di depth (da RealSense)
        "depth_min_mm": ..., "depth_max_mm": ...,  # soglie workspace (fase di grasp)
        "bbox": [x1, y1, x2, y2]      # opzionale (22/09/2026): crop X/Y attorno al target
    }
    """
    data = request.get_json()

    try:
        # Ricostruzione degli array numpy dai dati codificati in base64
        # (vedi decode_array() sopra).
        color = decode_array(data["color"], tuple(data["color_shape"]), data["color_dtype"])
        depth = decode_array(data["depth"], tuple(data["depth_shape"]), data["depth_dtype"])

        # Preparazione della point cloud (vedi build_point_cloud() sopra).
        end_points, cloud_points = build_point_cloud(
            color, depth,
            fx=data["fx"], fy=data["fy"], cx=data["cx"], cy=data["cy"],
            depth_scale=data["depth_scale"],
            depth_min_mm=data["depth_min_mm"],
            depth_max_mm=data["depth_max_mm"],
            bbox=data.get("bbox"),
        )

        # Inferenza + filtro collisioni.
        gg = run_inference(end_points)
        if COLLISION_THRESH > 0:
            gg = filter_collisions(gg, cloud_points)

        # Selezione del grasp migliore: non-maximum suppression (rimuove
        # grasp ridondanti/sovrapposti) poi ordinamento per punteggio.
        gg.nms()
        gg.sort_by_score()

        if len(gg) == 0:
            # Nessun grasp valido trovato (es. scena vuota, tutto filtrato
            # dal collision detector) — il client deve gestire questo caso,
            # non pubblicare un orientamento a caso.
            return jsonify({"success": False, "reason": "no_valid_grasps"}), 200

        best = gg[0]

        # Risposta: il grasp migliore (campi storici) piu' i primi TOP_K grasp
        # con posizione, tutti in camera frame -- il client sceglie quello
        # vicino al target, perche' il migliore in assoluto puo' essere su un
        # altro oggetto o sul tavolo.
        top_k = [
            {
                "score": float(g.score),
                "width": float(g.width),  # apertura della pinza richiesta [m]
                "rotation_matrix": g.rotation_matrix.tolist(),
                "translation": g.translation.tolist(),
            }
            for g in gg[:TOP_K]
        ]
        return jsonify({
            "success": True,
            "score": float(best.score),
            "rotation_matrix": best.rotation_matrix.tolist(),
            "translation": best.translation.tolist(),
            "grasps": top_k,
        }), 200

    except Exception as exc:
        # Qualunque errore nella pipeline (dati malformati, inferenza
        # fallita, ecc.) torna come errore esplicito al client invece di
        # far crashare il server — il nodo ROS2 dall'altra parte deve
        # poter loggare l'errore e non pubblicare un orientamento invalido.
        return jsonify({"success": False, "reason": str(exc)}), 500


if __name__ == "__main__":
    # host="0.0.0.0": ascolta su tutte le interfacce di rete della VM, non
    # solo localhost — necessario perche' le richieste arrivano dal laptop/
    # robot via VPN, non dalla VM stessa.
    # Porta scelta arbitrariamente (5001, non standard) — verificare che
    # non sia gia' occupata sulla VM, e che il firewall/VPN la lasci
    # passare (da testare insieme al test di connettivita' ROS2 gia'
    # segnato nei TODO).
    app.run(host="0.0.0.0", port=5001)