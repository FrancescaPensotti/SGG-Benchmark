import cv2
import numpy as np
import sys
import os

# ── Configurazione ──────────────────────────────────────────
# Parametri intrinseci della RealSense D435
# DA AGGIORNARE IN LAB con: ros2 topic echo /camera/camera/color/camera_info
# Cercare il campo K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
CAMERA_MATRIX = np.array([
    [615.0,   0.0, 320.0],
    [  0.0, 615.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)  # RealSense ha distorsione minima

# Dimensione del lato del marker ArUco in metri
# DA MISURARE sul marker stampato fisicamente
MARKER_SIZE = 0.05  # es. 5cm — aggiorna con la misura reale

# Dizionario ArUco da usare
ARUCO_DICT = cv2.aruco.DICT_6X6_250

# ── Setup ArUco ─────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def detect_aruco(frame):
    """
    Rileva marker ArUco nel frame e restituisce la loro posa.
    
    Returns:
        lista di dizionari con:
        - id: ID del marker
        - position: (x, y, z) in metri nel camera frame
        - z: profondità in metri (pseudo-depth)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    results = []
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            # Stima posa del marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i:i+1], MARKER_SIZE, CAMERA_MATRIX, DIST_COEFFS
            )
            
            x = tvec[0][0][0]  # sinistra/destra in metri
            y = tvec[0][0][1]  # su/giù in metri
            z = tvec[0][0][2]  # profondità in metri (pseudo-depth)
            
            results.append({
                'id': int(marker_id),
                'position': (x, y, z),
                'z': z
            })
            
            # Disegna marker e asse sul frame
            cv2.aruco.drawDetectedMarkers(frame, corners[i:i+1], ids[i:i+1])
            cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, MARKER_SIZE * 0.5)
            
            # Stampa posizione sul frame
            cv2.putText(frame, f"ID:{marker_id} z={z:.3f}m",
                       (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return results, frame


def pixel_to_meters_3d(cx_pixel, cy_pixel, z, camera_matrix):
    """
    Converte coordinate pixel in metri usando la profondità Z reale del marker ArUco.
    Sostituisce la conversione con scala fissa.
    
    Args:
        cx_pixel, cy_pixel: centroide oggetto in pixel
        z: profondità in metri dal marker ArUco
        camera_matrix: matrice intrinseca della camera
    
    Returns:
        (x_metri, y_metri): coordinate reali nel piano 2D
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx_cam = camera_matrix[0, 2]
    cy_cam = camera_matrix[1, 2]
    
    x_metri = (cx_pixel - cx_cam) * z / fx
    y_metri = (cy_pixel - cy_cam) * z / fy
    
    return (x_metri, y_metri)


# ── Test standalone con webcam ───────────────────────────────
if __name__ == '__main__':
    print("Avvio rilevamento ArUco — premi 'q' per uscire")
    print(f"Marker size: {MARKER_SIZE}m | Dizionario: DICT_6X6_250")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results, annotated = detect_aruco(frame)
        
        for r in results:
            print(f"Marker ID {r['id']}: pos=({r['position'][0]:.3f}, {r['position'][1]:.3f}, {r['position'][2]:.3f})m | z={r['z']:.3f}m")
        
        cv2.imshow("ArUco Detector", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()