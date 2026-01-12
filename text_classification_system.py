


# Text Classification System

# Text Classification System
#
# This script demonstrates a complete pipeline for classifying text messages as 'spam' or 'ham' (not spam)
# using machine learning. It uses the SMS Spam Collection Dataset.

import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# 1. Load Dataset
# The dataset contains SMS messages labeled as 'ham' (not spam) or 'spam'.
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
print('Sample data:')
print(df.head())

# 2. Preprocessing
def preprocess_text(text):
    """Lowercase, remove punctuation, tokenize, and remove stopwords."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

print('\nPreprocessing text...')
df['clean_text'] = df['text'].apply(preprocess_text)

# 3. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label'].map({'ham': 0, 'spam': 1})

# 4. Split Data
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training
# Train Multinomial Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# 6. Evaluation
def evaluate_model(model, X, y, model_name):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f'\n{model_name} Results:')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {prec:.4f}')
    print(f'Recall   : {rec:.4f}')
    print(f'F1-score : {f1:.4f}')
    return {'Model': model_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1}

results = []
results.append(evaluate_model(nb_model, X_test, y_test, 'Naive Bayes'))
results.append(evaluate_model(lr_model, X_test, y_test, 'Logistic Regression'))

# 7. Visualization
results_df = pd.DataFrame(results).set_index('Model')
results_df.plot(kind='bar', figsize=(8, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 8. Observations
print('\nObservations:')
print('- Both models achieve high accuracy on the SMS spam dataset.')
print('- Naive Bayes is fast and effective for text data with word frequencies.')
print('- Logistic Regression is a strong baseline and also performs very well.')
print('- Precision and recall are important:')
print('    - High precision: fewer false positives (ham marked as spam)')
print('    - High recall: fewer false negatives (spam missed)')

# 9. Extensions
print('\nPossible Extensions:')
print('- Add a web interface using Streamlit or Flask for real-time predictions.')
print('- Try more advanced models (SVM, Random Forest, or deep learning).')
print('- Experiment with more preprocessing (lemmatization, n-grams, etc.).')

# How to Run
# 1. Install required packages:
#    pip install numpy pandas scikit-learn nltk matplotlib seaborn
# 2. Run this script in your Python environment.

# Conclusion
# - This project demonstrates a complete text classification pipeline using Python and popular ML libraries.
# - Both Naive Bayes and Logistic Regression are suitable for text classification, with strong performance on spam detection.
# - The system can be extended with a web interface or more advanced models for further exploration.
