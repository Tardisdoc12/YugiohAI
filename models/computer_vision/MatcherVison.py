import os
import json
import cv2
import numpy as np
from collections import Counter, defaultdict
import faiss
from downloader_image import DownloaderImage


def get_images(
    path_to_stock_cards: str = "cards",
    path_file_with_url: str = "dataset/processed/yugioh_database_treated.json",
):
    downloader = DownloaderImage(path_to_stock_cards)
    path_file = path_file_with_url
    if not os.path.exists(path_file):
        raise "File doesn't exist"
    with open(path_file, "r") as file:
        cards: dict = json.load(file)

    for value in cards.values():
        path_card = value[1]
        downloader(path_card)

class MatcherVision:
    def __init__(self, nfeatures : int = 200, templates_dir : str = "cards"):
        # ORB init
        self.orb = cv2.ORB_create(nfeatures=nfeatures)
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # self.index = faiss.IndexHNSWFlat(d=32, M=32)
        quantizer = faiss.IndexFlatL2(32)
        self.index = faiss.IndexIVFPQ(quantizer, 32, 100, 8, 8)
        self.labels = []
        self.__load_index(templates_dir)
    
    def __load_index(self, template_dir):
        all_descriptors = []
        for filename in os.listdir(template_dir):
            path = os.path.join(template_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, des = self.orb.detectAndCompute(img, None)
            all_descriptors.append(des.astype(np.float32))
            self.labels.extend([filename] * des.shape[0])
        # Stack tous les descripteurs pour l'entraînement
        all_descriptors = np.vstack(all_descriptors)
        # Initialisation et entraînement de l'index
        d = all_descriptors.shape[1]
        nlist = 100
        m = 8
        nbits = 8
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        print("➡️ Entraînement de l'index...")
        self.index.train(all_descriptors)
        print("✅ Entraînement terminé. Ajout des vecteurs...")
        self.index.add(all_descriptors)

    def __call__(self, image_gray_to_compare : np.ndarray):
        _, des_img = self.orb.detectAndCompute(image_gray_to_compare, None)
        best_match = None
        _, indices = self.index.search(des_img.astype(np.float32), k=2)
        votes = [self.labels[i] for row in indices for i in row if i != -1]
        counter = Counter(votes)
        best_match, best_score = counter.most_common(1)[0]
        return best_match, best_score

    def batch_match(self, images_gray_list: list[np.ndarray]):
        descriptors_list = []
        image_indices = []

        # Extraire les descripteurs de chaque image
        for idx, img in enumerate(images_gray_list):
            _, des = self.orb.detectAndCompute(img, None)
            if des is not None:
                descriptors_list.append(des.astype(np.float32))
                image_indices.extend([idx] * des.shape[0])  # pour savoir à quelle image chaque descripteur appartient

        if not descriptors_list:
            return []

        # Concaténer tous les descripteurs
        batch_des = np.vstack(descriptors_list)

        # Rechercher dans FAISS
        _, indices = self.index.search(batch_des, k=2)

        # Associer votes par image
        votes_per_image = defaultdict(list)
        for des_idx, row in enumerate(indices):
            img_idx = image_indices[des_idx]
            for i in row:
                if i != -1:
                    votes_per_image[img_idx].append(self.labels[i])

        # Compter les votes pour chaque image
        results = []
        for img_idx in range(len(images_gray_list)):
            votes = votes_per_image[img_idx]
            if votes:
                counter = Counter(votes)
                best_match, best_score = counter.most_common(1)[0]
            else:
                best_match, best_score = None, 0
            results.append((best_match, best_score))
        
        return results


if __name__ == "__main__":
    import time
    matcher = MatcherVision(nfeatures=200)
    image = cv2.imread("dataset/raw/test_lv2_unicard.png",cv2.IMREAD_GRAYSCALE)
    image_2 = cv2.imread("dataset/raw/test_lv2_unicard_2.png",cv2.IMREAD_GRAYSCALE)
    image_3 = cv2.imread("dataset/raw/test_lv2_unicard_2.png",cv2.IMREAD_GRAYSCALE)
    start_time = time.time()
    best_match, best_score = matcher(image)
    print(f"Carte détectée : {best_match} score: {best_score}")
    best_match, best_score = matcher(image_2)
    print(f"le matching a pris:",time.time() - start_time)
    print(f"Carte détectée : {best_match} score: {best_score}")
    batch_images = [image, image_2]
    start_time = time.time()
    results = matcher.batch_match(batch_images)
    print(f"le matching a pris:",time.time() - start_time)
    for i, (match, score) in enumerate(results):
        print(f"Image {i}: carte = {match}, score = {score}")
