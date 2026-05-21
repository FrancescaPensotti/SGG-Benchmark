import cv2
import numpy as np
import sys
import os

# ── Configurazione ──────────────────────────────────────────
CAMERA_MATRIX = np.array([
    [912.123779296875,  0.0, 650.857666015625],
    [  0.0, 911.9319458007812, 383.904541015625],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

MARKER_SIZE = 0.03 # 3cm

ARUCO_DICT = cv2.aruco.DICT_6X6_250

# ── Setup ArUco ─────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Punti 3D del marker nel suo sistema di riferimento
OBJ_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

def detect_aruco(frame):
    """
    Rileva marker ArUco nel frame e restituisce la loro posa.

    Returns:
        results: lista di dizionari con id, position (x,y,z), z, corners
        frame: frame annotato
        corners: lista corners (vuota se nessun marker)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    results = []

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Stima posa con solvePnP (compatibile con OpenCV 4.x+)
            _, rvec, tvec = cv2.solvePnP(
                OBJ_POINTS, corners[i][0], CAMERA_MATRIX, DIST_COEFFS
            )
            tvec = tvec.reshape(1, 1, 3)

            x = tvec[0][0][0]
            y = tvec[0][0][1]
            z = tvec[0][0][2]

            results.append({
                'id': int(marker_id),
                'position': (x, y, z),
                'z': z,
                'corners': corners[i]
            })

            # Disegna marker e assi
            cv2.aruco.drawDetectedMarkers(frame, corners[i:i+1], ids[i:i+1])
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, MARKER_SIZE * 0.5)

            # Stampa z sul frame
            cv2.putText(frame, f"ID:{marker_id} z={z:.3f}m",
                        (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results, frame, corners if ids is not None else []


def pixel_to_meters_3d(cx_pixel, cy_pixel, z, camera_matrix):
    """
    Converte coordinate pixel in metri usando la profondità Z dal marker ArUco.
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx_cam = camera_matrix[0, 2]
    cy_cam = camera_matrix[1, 2]

    x_metri = (cx_pixel - cx_cam) * z / fx
    y_metri = (cy_pixel - cy_cam) * z / fy

    return (x_metri, y_metri)


def calcola_scala_da_aruco(corners, marker_size=MARKER_SIZE):
    """
    Calcola la scala pixel→metri dalle dimensioni note del marker ArUco.
    """
    # corners[0] ha shape (4, 2) — 4 angoli, ciascuno con (x, y)
    pts = corners[0]  # shape: (4, 2)
    larghezza_px = np.linalg.norm(pts[0] - pts[1])
    altezza_px = np.linalg.norm(pts[1] - pts[2])
    dimensione_px = (larghezza_px + altezza_px) / 2
    scala = marker_size / dimensione_px
    return scala

# ── Test standalone con webcam ───────────────────────────────
if __name__ == '__main__':
    print("Avvio rilevamento ArUco — premi 'q' per uscire")
    print(f"Marker size: {MARKER_SIZE}m | Dizionario: DICT_6X6_250")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results, annotated, _ = detect_aruco(frame)

        for r in results:
            print(f"Marker ID {r['id']}: pos=({r['position'][0]:.3f}, {r['position'][1]:.3f}, {r['position'][2]:.3f})m | z={r['z']:.3f}m")

        cv2.imshow("ArUco Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()