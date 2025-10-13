# preprocess_mtcnn.py
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# Crear detector MTCNN
detector = MTCNN()

# Directorios
INPUT_DIR = "data/subjects"
OUTPUT_DIR = "data/processed_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_and_save(image_path, output_path):
    """Detecta, recorta, alinea y normaliza un rostro con MTCNN."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo leer {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(img_rgb)

    if len(detections) == 0:
        print(f"No se detectó rostro en {image_path}")
        return

    # Tomamos el rostro con mayor confianza
    face = max(detections, key=lambda x: x['confidence'])
    x, y, w, h = face['box']

    # Recortar el rostro
    cropped_face = img_rgb[y:y+h, x:x+w]

    # Redimensionar a 160x160
    resized_face = cv2.resize(cropped_face, (160, 160))

    # Normalizar [0,1]
    normalized_face = resized_face / 255.0

    # Guardar
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor((normalized_face * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"Rostro procesado guardado en: {output_path}")

def process_all_images():
    """Procesa todas las imágenes de la carpeta de entrada."""
    for subject_folder in os.listdir(INPUT_DIR):
        subject_path = os.path.join(INPUT_DIR, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for img_name in os.listdir(subject_path):
            input_path = os.path.join(subject_path, img_name)
            output_folder = os.path.join(OUTPUT_DIR, subject_folder)
            output_path = os.path.join(output_folder, img_name)

            preprocess_and_save(input_path, output_path)

if __name__ == "__main__":
    process_all_images()
