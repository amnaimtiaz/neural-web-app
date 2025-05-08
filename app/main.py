from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from deepface import DeepFace
import numpy as np
import cv2
import os

from .emotion_utils import detect_emotion
from .face_utils import extract_face_embedding
from .demographics_utils import analyze_demographics
from .summarizer import summarize_text
from .image_classifiers import detect_mask, detect_blur, detect_beard

# Initialize FastAPI
app = FastAPI()

# Mount static files (uploaded images)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="app/templates")

# Function to analyze demographics (age, gender, race) using DeepFace
def analyze_demographics(file_path):
    try:
        result = DeepFace.analyze(file_path, actions=['age', 'gender', 'race'])
        age = result[0]['age']
        gender_dict = result[0]['gender']
        race = result[0]['dominant_race']
        
        dominant_gender = max(gender_dict, key=gender_dict.get)

        return {"age": age, "gender": dominant_gender, "race": race}, None
    except Exception as e:
        return None, str(e)
    
def detect_beard(image_path):
    try:
        analysis = DeepFace.analyze(image_path, actions=['gender', 'age'], enforce_detection=False)
        
        gender_dict = analysis[0]['gender']
        age = analysis[0]['age']

        # Extract dominant gender
        dominant_gender = max(gender_dict, key=gender_dict.get).lower()

        if dominant_gender == 'man' and age > 20:
            return "Possibly Beard"
        else:
            return "No Beard"
    except Exception as e:
        return f"Error in beard detection: {str(e)}"


# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# File upload and processing route
@app.post("/upload/") 
async def upload_file(request: Request, file: UploadFile = File(...), text: str = Form("")):
    # Save the uploaded file to the 'uploads' folder
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result_messages = []

    # Emotion Detection from Text
    if text.strip():  # Check if there is text input
        emotion, confidence = detect_emotion(text)
        result_messages.append(f"Emotion: {emotion} ({confidence}%)")

        # Add summarization
        summary = summarize_text(text)
        result_messages.append(f"Summary: {summary}")
        
    # Emotion Detection from Image
    try:
        image_emotion = DeepFace.analyze(file_path, actions=['emotion'], enforce_detection=False)
        emotion_from_image = image_emotion[0]['dominant_emotion']
        result_messages.append(f"Emotion (Image): {emotion_from_image}")
    except Exception as e:
        if isinstance(demographics['gender'], dict):
            dominant_gender = max(demographics['gender'], key=demographics['gender'].get)
            result_messages.append(f"Gender: {dominant_gender}")
        else:
            result_messages.append(f"Gender: {demographics['gender']}")
    

    # Face Recognition
    face_vector, face_error = extract_face_embedding(file_path)
    if face_vector is not None:
        result_messages.append("Face detected and encoded.")
    else:
        result_messages.append(f"Face Recognition: {face_error}")

    # Demographics (Age, Gender, Race using DeepFace)
    demographics, demo_error = analyze_demographics(file_path)
    if demographics:
        result_messages.append(f"Age: {demographics['age']}")
        result_messages.append(f"Gender: {demographics['gender']}")
        result_messages.append(f"Race: {demographics['race']}")
    else:
        result_messages.append(f"Demographics: {demo_error}")
        
    # Image Classifications (Mask, Blur, Beard)
    mask_status = detect_mask(file_path)
    result_messages.append(f"Mask: {mask_status}")
    
    blur_status = detect_blur(file_path)
    result_messages.append(f"Image: {blur_status}")
    
    beard_status = detect_beard(file_path)
    result_messages.append(f"Beard: {beard_status}")
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": "<br>".join(result_messages),
        "image_path": file_path,
        "text_input": text
    })
