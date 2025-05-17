import os
from pathlib import Path
from torch.utils.data import Dataset

class DatasetVision(Dataset):
    def __init__(self, path_to_dataset : str = "data_recognize/raw/vision_images"):
        self.images = []
        path_to_image_to_see = path_to_dataset
        for image in os.listdir(path_to_image_to_see):
            self.images.append(os.path.join(Path(path_to_image_to_see), Path(image)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
