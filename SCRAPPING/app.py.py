from flask import Flask, request, jsonify
from sentiment_analysis import sentiment_analysis
from summarization import summarize

app = Flask(__name__)

# Registering routes
app.register_blueprint(sentiment_analysis, url_prefix='/sentiment')
app.register_blueprint(summarize, url_prefix='/summarize')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
