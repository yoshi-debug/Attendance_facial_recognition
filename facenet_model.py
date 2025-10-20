# facenet_model.py
from tensorflow.keras import layers, models, backend as K
import tensorflow as tf
import numpy as np
from PIL import Image

def create_facenet_model():
    inputs = layers.Input(shape=(160, 160, 3))

    x = layers.Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3,3), strides=2)(x)
    
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128)(x)  # Embedding size 128
    x = layers.Lambda(lambda t: K.l2_normalize(t, axis=1))(x)

    model = models.Model(inputs, x, name='FaceNet')
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize((160,160))
    img = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

def get_embedding(model, image_path):
    img = preprocess_image(image_path)
    embedding = model.predict(img)
    return embedding[0]
