from ultralytics import YOLOWorld

# Carica il modello YOLO-World
model = YOLOWorld("yolov8s-world.pt")

# Definisci gli oggetti che vuoi trovare con testo libero
model.set_classes(["chair","person","cup","laptop"])

# Esegui su un'immagine
results = model("demo/example.jpg")

# Stampa cosa ha trovato
for result in results:
    for box in result.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls]
        print(f"Trovato: {label} con confidenza {conf:.2f}")

# Salva l'immagine con le bounding box
results[0].save("demo/yoloworld_result.jpg")
print("Immagine salvata in demo/yoloworld_result.jpg")