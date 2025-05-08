from deepface import DeepFace

# Function to analyze demographics (age, gender, race) using DeepFace
def analyze_demographics(file_path):
    try:
        # DeepFace will automatically download the necessary models
        result = DeepFace.analyze(file_path, actions=['age', 'gender', 'race'])

        # Extract the required information
        age = result[0]['age']
        gender = result[0]['gender']
        race = result[0]['dominant_race']

        return {"age": age, "gender": gender, "race": race}, None
    except Exception as e:
        return None, str(e)
