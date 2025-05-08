from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

# Get emotion labels from config
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def detect_emotion(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=1)
    top_idx = torch.argmax(probs, dim=1).item()
    emotion = labels[top_idx]
    confidence = probs[0][top_idx].item()
    return emotion, round(confidence * 100, 2)
