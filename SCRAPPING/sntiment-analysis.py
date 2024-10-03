from flask import Blueprint, request, jsonify
from transformers import BartTokenizer, BartForSequenceClassification

sentiment_analysis = Blueprint('sentiment_analysis', __name__)

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForSequenceClassification.from_pretrained('facebook/bart-large')

@sentiment_analysis.route('/', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    url = data['url']
    
    # You would fetch the article content here
    text = "Your article text here"

    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax().item()

    sentiment = "positive" if predicted_class_idx == 2 else "negative" if predicted_class_idx == 0 else "neutral"
    return jsonify({'sentiment': sentiment})
