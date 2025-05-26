################################################################################
# filename: create_database.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

import os
import json
import requests
from globals import api_url

################################################################################

def create_database_file(path_file: str) -> None:
    if os.path.exists(path_file):
        return
    reponse = requests.get(api_url)
    with open(path_file, "w") as file:
        json.dump(reponse.json(), file)

################################################################################

def treat_database_to_image(path_to_database):
    file_path_final = transform_path(path_to_database)
    if os.path.exists(file_path_final):
        return

    with open(path_to_database, "r") as file:
        datas = json.load(file)

    cards = {}
    for card in datas["data"]:
        id_card = card["card_images"][0]["id"]
        name_card = card["name"]
        for image_id in range(len(card["card_images"])):
            image_card_path = card["card_images"][image_id]["image_url_small"]
            id_images = image_card_path.split("/")[-1].split(".")[0]
            if id_card != id_images:
                cards[id_images] = [name_card, image_card_path]
            else:
                cards[id_card] = [name_card, image_card_path]

    with open(file_path_final, "w") as file:
        json.dump(cards, file)

################################################################################

def transform_path(path_to_treat: str) -> str:
    path_list = path_to_treat.split(".")
    return ".".join([path_list[0] + "_treated", path_list[-1]])

################################################################################

def main():
    
    path_file = "data_recognize/processed/yugioh_database.json"
    os.makedirs("data_recognize/processed",exist_ok=True)
    create_database_file(path_file)
    treat_database_to_image(path_file)
    with open(transform_path(path_file), "r") as file:
        print(len(json.load(file)))

################################################################################

if __name__ == "__main__":
    main()

################################################################################
# End of File
################################################################################