################################################################################
# filename: yolo_vision.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import cv2
import numpy as np
import onnxruntime as ort
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


################################################################################

def inference_global(img_path):
    from concurrent.futures import ThreadPoolExecutor
    yolo = YOLO("runs/detect/yolov8n_custom7/weights/best.pt").to("cuda")
    matcher = MatcherVision(nfeatures=500)
    
    original_img = cv2.imread(img_path)

    start_time = time.time()

    results = yolo.predict(source=img_path, save=False,device="cuda",verbose=False)
    temp_int = time.time()
    print(f"la detection a pris:",temp_int - start_time)

    images= []
    # Extraire les résultats
    boxes = results[0].boxes
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])  # classe prédite
        if cls == 0:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            crop = original_img[y1:y2, x1:x2]  # découper l'image
            cv2.imwrite(f"output/detections_class0/detection_{i}.png", crop)
            images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    trait_time = time.time()
    print(f"le traitement a pris:",trait_time - temp_int)
    with ThreadPoolExecutor() as executor:
        best_match = list(executor.map(matcher, images))
    final_time = time.time()
    print(f"le matching a pris:",final_time - trait_time)
    print("au global : ",final_time - start_time)
    for i, (match, score) in enumerate(best_match):
        print(f"Image {i}: carte = {match}, score = {score}")

if __name__ == "__main__":
    # train_yolo_vision()
    img_path = "data_recognize/raw/field.png"
    inference_global(img_path)

################################################################################
# End of File
################################################################################