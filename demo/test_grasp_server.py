import numpy as np
import base64
import requests

# Dati finti (rumore casuale) solo per verificare che la pipeline non
# crashi end-to-end — non aspettarti un grasp fisicamente sensato.
H, W = 480, 640
color = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
depth = np.random.uniform(300, 500, (H, W)).astype(np.float32)  # mm, range plausibile

def encode(array):
    return base64.b64encode(array.tobytes()).decode("utf-8")

payload = {
    "color": encode(color), "color_shape": list(color.shape), "color_dtype": "uint8",
    "depth": encode(depth), "depth_shape": list(depth.shape), "depth_dtype": "float32",
    "fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0,
    "depth_scale": 0.001,  # 1mm per unita', tipico RealSense
    "depth_min_mm": 300, "depth_max_mm": 500,
}

resp = requests.post("http://127.0.0.1:5001/predict_grasp", json=payload)
print("Status:", resp.status_code)
print("Body:", resp.json())