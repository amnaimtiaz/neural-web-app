from keras_facenet import FaceNet
import numpy as np
from PIL import Image
import cv2
import os

embedder = FaceNet()

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))
    return np.asarray(img)

def get_face_embedding(image_path):
    img_array = preprocess_image(image_path)
    embeddings = embedder.embeddings([img_array])
    return embeddings[0]

def load_known_faces(known_dir="known_faces"):
    known_embeddings = []
    known_labels = []
    for file in os.listdir(known_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            label = file.split('.')[0]
            path = os.path.join(known_dir, file)
            embedding = get_face_embedding(path)
            known_embeddings.append(embedding)
            known_labels.append(label)
    return known_embeddings, known_labels

def recognize_face(face_path="detected_face.jpg", known_dir="known_faces"):
    known_embeddings, known_labels = load_known_faces(known_dir)
    target_embedding = get_face_embedding(face_path)

    # Compare using cosine similarity
    similarities = np.dot(known_embeddings, target_embedding) / (
        np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(target_embedding)
    )

    best_match_idx = np.argmax(similarities)
    if similarities[best_match_idx] > 0.7:  # Threshold
        return known_labels[best_match_idx]
    else:
        return "Unknown"
