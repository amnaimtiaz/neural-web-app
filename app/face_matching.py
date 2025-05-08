import os
import face_recognition
import pickle

FACE_DB_DIR = "app/face_db"
EMBEDDINGS_FILE = "app/face_db/embeddings.pkl"

def build_face_db():
    known_encodings = []
    known_names = []

    for filename in os.listdir(FACE_DB_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(FACE_DB_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_names.append(name)

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump((known_names, known_encodings), f)

def find_face_match(upload_path):
    with open(EMBEDDINGS_FILE, "rb") as f:
        known_names, known_encodings = pickle.load(f)

    unknown_image = face_recognition.load_image_file(upload_path)
    unknown_encodings = face_recognition.face_encodings(unknown_image)

    if not unknown_encodings:
        return "No face found in uploaded image."

    for unknown_encoding in unknown_encodings:
        results = face_recognition.compare_faces(known_encodings, unknown_encoding)
        distances = face_recognition.face_distance(known_encodings, unknown_encoding)

        best_match_index = distances.argmin()
        if results[best_match_index]:
            return f"Matched: {known_names[best_match_index]}"

    return "No Match Found"
