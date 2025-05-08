import face_recognition
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
import os

# Load pre-trained Vision Transformer
extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_face_embedding(image_path):
    # Load image using face_recognition
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return None, "No face detected."

    # Take the first detected face
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)

    # Preprocess for ViT
    inputs = extractor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token as embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return embedding, None
