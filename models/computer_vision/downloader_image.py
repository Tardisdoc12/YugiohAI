################################################################################
# filename: downloader_image.py
# Author: Jean Anquetil
# Email: janquetil@e-vitech.com
# Date: 15/05,2025
################################################################################

import os
import requests

################################################################################

class DownloaderImage:
    def __init__(self, path_to_images):
        self.path_to_images = path_to_images
        if not os.path.exists(self.path_to_images):
            os.makedirs(self.path_to_images)

    def __call__(self, url_path):
        image_local_path = self.path_to_images + "/" + url_path.split("/")[-1]
        if os.path.exists(image_local_path):
            return

        response = requests.get(url_path)
        if response.status_code == 200:
            with open(image_local_path, "wb") as f:
                f.write(response.content)
        else:
            raise f"Erreur ({response.status_code}) lors du téléchargement : {url_path}"

################################################################################
# End of File
################################################################################