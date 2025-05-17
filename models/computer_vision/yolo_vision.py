################################################################################
# filename: yolo_vision.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from ultralytics import YOLO
from MatcherVison import MatcherVision

################################################################################
def train_yolo_vision():
    yolo = YOLO("training/yolov8n.pt")
    yolo.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        workers=4,
        name="yolov8n_custom" 
    )

if __name__ == "__main__":
    # train_yolo_vision()
    import torch
    print("GPU disponible :", torch.cuda.is_available())
    import cv2
    yolo = YOLO("runs/detect/yolov8n_custom7/weights/best.pt")
    print("Device utilisé par YOLO :", yolo.device)
    matcher = MatcherVision(nfeatures=500)
    img_path = "data_recognize/raw/field.png"
    original_img = cv2.imread(img_path)
    start_time = time.time()
    results = yolo.predict(source=img_path, save=False)

    images= []
    # Extraire les résultats
    boxes = results[0].boxes
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])  # classe prédite
        if cls == 0:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            print("x1:", x1, "y1:", y1)
            crop = original_img[y1:y2, x1:x2]  # découper l'image
            images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

    best_match = matcher.batch_match(images)
    print(f"le matching a pris:",time.time() - start_time)
    for i, (match, score) in enumerate(best_match):
        print(f"Image {i}: carte = {match}, score = {score}")

################################################################################
# End of File
################################################################################