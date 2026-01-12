# Add topic classification model/vectorizer/labels
TOPIC_MODEL_PATH = 'model_topic.pkl'
TOPIC_VECTORIZER_PATH = 'vectorizer_topic.pkl'
TOPIC_LABELS_PATH = 'topic_label_names.pkl'
import os
import pickle
import re
import string
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template

nltk.download('stopwords')

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')

# Load or train models and vectorizer

# Model and vectorizer paths
SPAM_MODEL_PATH = 'model_spam.pkl'
SPAM_VECTORIZER_PATH = 'vectorizer_spam.pkl'
SENT_MODEL_PATH = 'model_sentiment.pkl'
SENT_VECTORIZER_PATH = 'vectorizer_sentiment.pkl'

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


# Load models and vectorizers
# Load models and vectorizers
with open(SPAM_MODEL_PATH, 'rb') as f:
    spam_model = pickle.load(f)
with open(SPAM_VECTORIZER_PATH, 'rb') as f:
    spam_vectorizer = pickle.load(f)
with open(SENT_MODEL_PATH, 'rb') as f:
    sent_model = pickle.load(f)
with open(SENT_VECTORIZER_PATH, 'rb') as f:
    sent_vectorizer = pickle.load(f)
# Topic classification
with open(TOPIC_MODEL_PATH, 'rb') as f:
    topic_model = pickle.load(f)
with open(TOPIC_VECTORIZER_PATH, 'rb') as f:
    topic_vectorizer = pickle.load(f)
with open(TOPIC_LABELS_PATH, 'rb') as f:
    topic_label_names = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html', prediction=None, model=None, input_text=None, task=None)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    task = request.form.get('task', 'fraud')
    clean_text = preprocess_text(text)
    if task == 'fraud':
        X = spam_vectorizer.transform([clean_text])
        pred = spam_model.predict(X)[0]
        label = 'Spam' if pred == 1 else 'Ham'
        model_name = 'Fraud Detection'
    elif task == 'sentiment':
        X = sent_vectorizer.transform([clean_text])
        pred = sent_model.predict(X)[0]
        label = pred.capitalize()
        model_name = 'Sentiment Analysis'
    elif task == 'topic':
        X = topic_vectorizer.transform([clean_text])
        pred = topic_model.predict(X)[0]
        label = topic_label_names[pred]
        model_name = 'Topic Classification'
    else:
        label = 'Unknown'
        model_name = 'Unknown'
    return render_template('index.html', prediction=label, model=model_name, input_text=text, task=task)

if __name__ == '__main__':
    app.run(debug=True)
