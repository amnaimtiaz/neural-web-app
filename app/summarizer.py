from transformers import pipeline

# Load once to reuse
summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text.split()) < 30:
        return "Text too short to summarize. Enter a paragraph of at least 30 words."
    summary = summarizer_pipeline(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']
