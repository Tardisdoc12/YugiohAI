################################################################################
# filename: yolo_vision.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from ultralytics import YOLO
from MatcherVison import MatcherVision

################################################################################

def train_yolo_vision():
    yolo = YOLO("training/yolo11n.pt")
    yolo.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=16,
        workers=4,
        name="yolov11n_custom",
        device="cuda"
    )

################################################################################

def get_yolo_model() -> YOLO:
    yolo = YOLO("runs/detect/yolov8n_custom7/weights/best.pt").to("cuda")
    return yolo

################################################################################

def warmup_yolo(yolo_model : YOLO) -> None:
    for _ in range(5):
        yolo_model.predict(source="data_recognize/raw/field.png", save=False, device="cuda",verbose=False)

################################################################################

def inference_global(img_path):
    from concurrent.futures import ThreadPoolExecutor
    yolo = YOLO("runs/detect/yolov8n_custom7/weights/best.pt").to("cuda")
    matcher = MatcherVision(nfeatures=370)
    yolo.predict(source=img_path, save=False,device="cuda",verbose=False)
    original_img = cv2.imread(img_path)

    results = yolo.predict(source=img_path, save=False,device="cuda",verbose=False)
    images= []
    # Extraire les résultats
    boxes = results[0].boxes
    os.makedirs("output/detections_class0",exist_ok=True)
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])  # classe prédite
        print(cls)
        if cls == 0 and boxes.conf[0] > 0.37:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            crop = original_img[y1:y2, x1:x2]  # découper l'image
            os.makedirs(f"output/detections_class0",exist_ok=True)
            cv2.imwrite(f"output/detections_class0/detection_{i}.png",crop)
            images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    with ThreadPoolExecutor() as executor:
        best_match = list(executor.map(matcher, images))
    for i, (match, score) in enumerate(best_match):
        print(f"Image {i}: carte = {match}, score = {score}")

################################################################################

if __name__ == "__main__":
    # train_yolo_vision()
    img_path = "data_recognize/raw/choice_side_bar.png"
    inference_global(img_path)

################################################################################
# End of File
################################################################################