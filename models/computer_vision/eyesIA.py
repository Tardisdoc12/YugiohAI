#!/usr/bin/env python
# encoding: utf-8
################################################################################
# author : Jean Anquetil
# Created on : 19
# nom du fichier : eyesIA.py
################################################################################

import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from MatcherVison import (
    MatcherVision
)
from yolo_vision import (
    train_yolo_vision,
    get_yolo_model,
    warmup_yolo
)

################################################################################

class EyesComputerVision:
    def __init__(self):
        self.matcher = MatcherVision(370)
        if os.path.exists("runs/yolov11n_custom/weights/best.pt"):
            train_yolo_vision()
        self.yolo = get_yolo_model()
        warmup_yolo(self.yolo)
        self.images = []
    
    def getCardsInImage(self,image_path) -> None:
        original_img = cv2.imread(image_path)
        results = self.yolo.predict(source=image_path, save=True,device="cuda",verbose=False)
        boxes = results[0].boxes
        counting_face_down = 0
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if cls == 1:
                counting_face_down += 1
            elif cls == 0:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)  # coordonnées (x1, y1, x2, y2)
                x1, y1, x2, y2 = xyxy
                crop = original_img[y1:y2, x1:x2]  # découper l'image
                self.images.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        counting_face_down = max(counting_face_down - 4, 0)
        print(f"Il y a :\n{counting_face_down} cartes face cachée\net\n{len(self.images)} cartes visibles sur le terrain")
    
    def get_name_cards(self):
        with ThreadPoolExecutor() as executor:
            best_match = list(executor.map(self.matcher, self.images))
        for i, (match, score) in enumerate(best_match):
            if score > 1:
                print(f"Image {i}: carte = {match}, score = {score}")

    def reset_images_cards(self):
        self.images = []

################################################################################

if __name__ == "__main__":
    import time
    eyes = EyesComputerVision()
    start_time = time.time()
    eyes.getCardsInImage("data_recognize/raw/test_lv2.png")
    int_time = time.time()
    eyes.get_name_cards()
    end_time = time.time()
    print("on a l'inference + post traitement en :",int_time - start_time)
    print("le matching est en :",end_time - int_time)
    print("Au total on en a eu pour :",end_time - start_time)

################################################################################
# End Of File
################################################################################