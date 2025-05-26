#!/usr/bin/env python
# encoding: utf-8
################################################################################
# author : Jean Anquetil
# Created on : 19
# nom du fichier : eyesIA.py
################################################################################

import os
import json
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

dict_zones = {
    "hand":[980,1080],
    "spell&trap":[770,980],
    "monster":[580, 770],
    "opp_monster":[230,410],
    "opp_spell&trap":[110,230],
    "opp_hand":[0,110]
}

################################################################################

class EyesComputerVision:
    def __init__(self):
        self.matcher = MatcherVision(500)
        if os.path.exists("runs/yolov8n_custom7/weights/best.pt"):
            train_yolo_vision()
        self.yolo = get_yolo_model()
        warmup_yolo(self.yolo)
        self.images = []
        self.original_img = None
        self.cards = {"hand":{},"spell&trap":{},"monster":{},"opp_monster":{},"opp_spell&trap":{},"opp_hand":{}}
        with open("data_recognize/processed/yugioh_database_treated.json","r") as file:
            self.debug_code = json.load(file)
    
    def __getCardFromBox(self,box):
        counting_face_down = 0
        cls = int(box.cls[0])
        if cls == 1:
            counting_face_down += 1
        elif cls == 0:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)  # coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = xyxy
            crop = self.original_img[y1:y2, x1:x2]
            match,score = self.matcher(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY))
            if score > 1:
                coord = box.xywh[0].cpu().numpy().astype(int)[:2]
                name = self.debug_code[match.split(".")[0]][0]
                for zone in dict_zones:
                    if coord[1] >= dict_zones[zone][0] and coord[1] <= dict_zones[zone][1]:
                        self.cards[zone][name] = coord
                # if coord[1] >= 980:
                #     self.cards["hand"][name] = box.xywh[0].cpu().numpy().astype(int)[:2]
                # else:
                #     self.cards["field"][name] = box.xywh[0].cpu().numpy().astype(int)[:2]
        else:
            print("action")

    def getCardsInImage(self,image_path) -> None:
        self.original_img = cv2.imread(image_path)
        results = self.yolo.predict(source=image_path, save=False,device="cuda",verbose=False)
        boxes = results[0].boxes
        # for box in boxes:
        #     self.__getCardFromBox(box)
        with ThreadPoolExecutor() as executor:
            executor.map(self.__getCardFromBox,boxes)
        print(self.cards)
    
    def get_name_cards(self):
        with ThreadPoolExecutor() as executor:
            best_match = list(executor.map(self.matcher, self.images))
        for i, (match, score) in enumerate(best_match):
            if score > 1:
                name = self.debug_code[match.split(".")[0]][0]
                print(f"Image {i}: carte = {name}, score = {score}")

    def reset_images_cards(self):
        self.images = []

################################################################################

if __name__ == "__main__":
    import time
    eyes = EyesComputerVision()
    start_time = time.time()
    eyes.getCardsInImage("data_recognize/raw/test_lv2.png")
    int_time = time.time()
    print("on a l'inference + post traitement en :",int_time - start_time)

################################################################################
# End Of File
################################################################################