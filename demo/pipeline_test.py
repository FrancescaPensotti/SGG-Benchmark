from ultralytics import YOLOWorld
import cv2

# 1. Carica YOLO-World
model = YOLOWorld("yolov8s-world.pt")

# 2. Definisci gli oggetti da cercare
model.set_classes(["laptop", "cup", "phone", "book", "keyboard", "mouse"])

# 3. Apri la webcam
cap = cv2.VideoCapture(0)

print("Webcam aperta. Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. YOLO-World rileva gli oggetti
    results = model(frame, verbose=False)
    
    # 5. Stampa gli oggetti trovati
    objects_found = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls]
            if conf > 0.3:
                objects_found.append(label)
    
    if objects_found:
        print(f"Oggetti nella scena: {objects_found}")

    # 6. Mostra il frame con le bounding box
    annotated = results[0].plot()
    cv2.imshow("YOLO-World Live", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()