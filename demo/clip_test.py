import torch
import clip
import cv2
import numpy as np
from ultralytics import YOLOWorld
from PIL import Image

# 1. Carica YOLO-World e CLIP
print("Carico i modelli...")
yolo = YOLOWorld("yolov8s-world.pt")
yolo.set_classes(["laptop", "cup", "phone", "book", "keyboard", "mouse", "sunglasses", "keys", "wallet"])

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
print(f"Modelli caricati! Uso: {device}")

# 2. Apri la webcam
cap = cv2.VideoCapture(0)
print("Webcam aperta. Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLO-World rileva gli oggetti
    results = yolo(frame, verbose=False)
    
    objects_embeddings = {}
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf < 0.3:
                continue
                
            cls = int(box.cls)
            label = yolo.names[cls]
            
            # 4. Ritaglia l'oggetto dall'immagine
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # 5. Calcola embedding CLIP del ritaglio
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            crop_tensor = clip_preprocess(crop_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = clip_model.encode_image(crop_tensor)
                embedding = embedding / embedding.norm()  # normalizza
            
            objects_embeddings[label] = embedding
            print(f"  {label} (conf: {conf:.2f}) → embedding shape: {embedding.shape}")
    
    # 6. Calcola similarità tra oggetti trovati
    if len(objects_embeddings) >= 2:
        labels = list(objects_embeddings.keys())
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                sim = torch.cosine_similarity(
                    objects_embeddings[labels[i]], 
                    objects_embeddings[labels[j]]
                ).item()
                print(f"  Similarità {labels[i]} ↔ {labels[j]}: {sim:.3f}")

    # Mostra frame
    annotated = results[0].plot()
    cv2.imshow("YOLO-World + CLIP", annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()