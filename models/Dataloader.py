import cv2
import easyocr

reader = easyocr.Reader(["en"])
result = reader.readtext("dataset/field.png")
# print(result)
# import pytesseract


# # Facultatif : si tesseract n'est pas dans le PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Charger l'image avec OpenCV
image = cv2.imread("dataset/field.png")

# Convertir en niveaux de gris (améliore souvent la détection)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # OCR avec informations de position
# data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

# Parcourir les résultats
for data in result:
    text = data[1].strip()
    conf = data[2]
    if text != "" and conf > 0.50:
        x_min, x_max, y_min, y_max = (
            tuple(data[0][0]),
            tuple(data[0][1]),
            tuple(data[0][2]),
            tuple(data[0][3]),
        )
        # Dessiner le rectangle
        cv2.rectangle(
            image,
            (int(x_min[0]), int(x_min[1])),
            (int(y_max[0]), int(y_max[1])),
            (0, 255, 0),
            2,
        )
        # Afficher le texte
        cv2.putText(
            image,
            text,
            (int(x_min[0]), int(x_min[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

# Sauvegarder ou afficher le résultat
cv2.imwrite("output_2.png", image)
