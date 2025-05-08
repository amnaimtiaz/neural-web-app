from app.face_matching import find_face_match

# Replace with the path to your test image
query_image_path = "test_images/valorie_brabazon.jpg"

result = find_face_match(query_image_path)
print("🔍 Match result:", result)
