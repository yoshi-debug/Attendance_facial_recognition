# mtcnn_preprocess.py
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image

detector = MTCNN()

RAW_DIR = "data/raw_faces"
PROCESSED_DIR = "data/processed_faces"

def align_face(image, left_eye, right_eye):
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(np.mean([left_eye, right_eye], axis=0).astype(int))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))
    return aligned

def preprocess_faces():
    for person in os.listdir(RAW_DIR):
        person_dir = os.path.join(RAW_DIR, person)
        output_dir = os.path.join(PROCESSED_DIR, person)
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(img)

            if not detections:
                continue

            for i, det in enumerate(detections):
                x, y, w, h = det['box']
                keypoints = det['keypoints']
                face = img[y:y+h, x:x+w]
                face = align_face(face, keypoints['left_eye'], keypoints['right_eye'])
                face = Image.fromarray(face).resize((160, 160))
                face.save(os.path.join(output_dir, f"{person}_{i}.jpg"))

        print(f"✅ Procesadas imágenes para {person}")

if __name__ == "__main__":
    preprocess_faces()
