import cv2
from deepface import DeepFace
import numpy as np
import os

def detect_mask(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        # Simple logic: if mouth/nose covered, low emotion clarity, might be wearing mask
        dominant_emotion = result[0]['dominant_emotion']
        if dominant_emotion == "neutral":
            return "Possibly Wearing Mask"
        return "No Mask"
    except:
        return "Detection Failed"

def detect_blur(image_path, threshold=100.0):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "Invalid image"
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return "Blurred" if laplacian_var < threshold else "Clear"

def detect_beard(image_path):
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi_color = image[y:y+h, x:x+w]
            lower_half = roi_color[h//2:, :]  # Beard is usually in lower half
            darkness = np.mean(lower_half)
            if darkness < 110:
                return "Beard"
        return "No Beard"
    except:
        return "Error detecting beard"
