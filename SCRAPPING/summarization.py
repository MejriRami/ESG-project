from flask import Blueprint, request, jsonify
from transformers import BartTokenizer, BartForConditionalGeneration

summarization = Blueprint('summarization', __name__)

# Load the BART model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

@summarization.route('/', methods=['POST'])
def summarize():
    data = request.get_json()
    url = data['url']
    
    # You would fetch the article content here
    text = "Your article text here"

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(**inputs, min_length=50, max_length=150)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({'summary': summary})
