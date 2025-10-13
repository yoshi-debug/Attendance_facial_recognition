# facenet_model.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Carpeta con las imágenes preprocesadas
PROCESSED_DIR = "data/processed_faces"

def load_preprocessed_faces():
    """Carga las imágenes preprocesadas (160x160x3) listas para FaceNet."""
    faces = []
    labels = []

    for subject_folder in os.listdir(PROCESSED_DIR):
        subject_path = os.path.join(PROCESSED_DIR, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for img_name in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Convertir a RGB y normalizar
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype('float32') / 255.0

            faces.append(img_rgb)
            labels.append(subject_folder)

    return np.array(faces), labels


if __name__ == "__main__":
    X, y = load_preprocessed_faces()
    print(f"Total de rostros cargados: {len(X)}")
    if len(X) > 0:
        print(f"Forma de una imagen: {X[0].shape}")
        print(f"Ejemplo de etiqueta: {y[0]}")
