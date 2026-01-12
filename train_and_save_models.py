import os
import pandas as pd
import re
import string
import nltk
import pickle
import joblib
import requests
from nltk.corpus import stopwords, movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups

nltk.download('stopwords')
nltk.download('movie_reviews')

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#############################
# --- Topic Classification Model (20 Newsgroups) ---
#############################
print('Downloading and saving topic classification dataset...')
topic_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
joblib.dump(topic_data, os.path.join(DATA_DIR, 'topic_20newsgroups.pkl'))
topic_texts = topic_data.data
topic_labels = topic_data.target
topic_label_names = topic_data.target_names
topic_df = pd.DataFrame({'text': topic_texts, 'label': topic_labels})
topic_df = topic_df.dropna()
topic_df['clean_text'] = topic_df['text'].apply(preprocess_text)
topic_vectorizer = TfidfVectorizer(max_features=3000)
X_topic = topic_vectorizer.fit_transform(topic_df['clean_text'])
y_topic = topic_df['label']
print('Topic class distribution:')
print(pd.Series(y_topic).value_counts())
X_train_topic, X_test_topic, y_train_topic, y_test_topic = train_test_split(
    X_topic, y_topic, test_size=0.2, random_state=42, stratify=y_topic)
topic_model = LogisticRegression(max_iter=1000)
topic_model.fit(X_train_topic, y_train_topic)
with open('model_topic.pkl', 'wb') as f:
    pickle.dump(topic_model, f)
with open('vectorizer_topic.pkl', 'wb') as f:
    pickle.dump(topic_vectorizer, f)
with open('topic_label_names.pkl', 'wb') as f:
    pickle.dump(topic_label_names, f)

#############################
# --- Fraud (Spam) Detection Model ---
#############################
print('Downloading and saving spam dataset...')
spam_url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
spam_path = os.path.join(DATA_DIR, 'sms.tsv')
if not os.path.exists(spam_path):
    r = requests.get(spam_url)
    with open(spam_path, 'wb') as f:
        f.write(r.content)
spam_df = pd.read_csv(spam_path, sep='\t', header=None, names=['label', 'text'])
spam_df['clean_text'] = spam_df['text'].apply(preprocess_text)
spam_vectorizer = TfidfVectorizer()
X_spam = spam_vectorizer.fit_transform(spam_df['clean_text'])
y_spam = spam_df['label'].map({'ham': 0, 'spam': 1})
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42, stratify=y_spam)
spam_model = MultinomialNB()
spam_model.fit(X_train_s, y_train_s)
with open('model_spam.pkl', 'wb') as f:
    pickle.dump(spam_model, f)
with open('vectorizer_spam.pkl', 'wb') as f:
    pickle.dump(spam_vectorizer, f)

#############################
# --- Sentiment Analysis Model (using NLTK movie_reviews) ---
#############################
print('Saving NLTK movie_reviews dataset...')
sent_texts = []
sent_labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        sent_texts.append(' '.join(movie_reviews.words(fileid)))
        sent_labels.append(category)
sent_df = pd.DataFrame({'text': sent_texts, 'label': sent_labels})
joblib.dump(sent_df, os.path.join(DATA_DIR, 'movie_reviews_df.pkl'))
sent_df['clean_text'] = sent_df['text'].apply(preprocess_text)
sent_vectorizer = TfidfVectorizer()
X_sent = sent_vectorizer.fit_transform(sent_df['clean_text'])
y_sent = sent_df['label']
print('Sentiment class distribution:')
print(y_sent.value_counts())
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42, stratify=y_sent)
sent_model = LogisticRegression(max_iter=1000)
sent_model.fit(X_train_t, y_train_t)
with open('model_sentiment.pkl', 'wb') as f:
    pickle.dump(sent_model, f)
with open('vectorizer_sentiment.pkl', 'wb') as f:
    pickle.dump(sent_vectorizer, f)

print('All models and datasets saved successfully.')
topic_df = topic_df.dropna()
topic_df['clean_text'] = topic_df['text'].apply(preprocess_text)
topic_vectorizer = TfidfVectorizer(max_features=3000)
X_topic = topic_vectorizer.fit_transform(topic_df['clean_text'])
nltk.download('stopwords')
nltk.download('stopwords')

import pandas as pd
import re
import string
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#############################
# --- Topic Classification Model (20 Newsgroups) ---
#############################
print('Downloading and preparing topic classification dataset...')
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
topic_texts = newsgroups.data
topic_labels = newsgroups.target
topic_label_names = newsgroups.target_names

topic_df = pd.DataFrame({'text': topic_texts, 'label': topic_labels})
topic_df = topic_df.dropna()
topic_df['clean_text'] = topic_df['text'].apply(preprocess_text)
topic_vectorizer = TfidfVectorizer(max_features=3000)
X_topic = topic_vectorizer.fit_transform(topic_df['clean_text'])
y_topic = topic_df['label']

# Print class distribution
print('Topic class distribution:')
print(pd.Series(y_topic).value_counts())

X_train_topic, X_test_topic, y_train_topic, y_test_topic = train_test_split(
    X_topic, y_topic, test_size=0.2, random_state=42, stratify=y_topic)
topic_model = LogisticRegression(max_iter=1000)
topic_model.fit(X_train_topic, y_train_topic)
with open('model_topic.pkl', 'wb') as f:
    pickle.dump(topic_model, f)
with open('vectorizer_topic.pkl', 'wb') as f:
    pickle.dump(topic_vectorizer, f)
with open('topic_label_names.pkl', 'wb') as f:
    pickle.dump(topic_label_names, f)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

#############################
# --- Topic Classification Model (20 Newsgroups) ---
#############################
print('Downloading and preparing topic classification dataset...')
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
topic_texts = newsgroups.data
topic_labels = newsgroups.target
topic_label_names = newsgroups.target_names


nltk.download('stopwords')




# --- Fraud (Spam) Detection Model ---
spam_url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
spam_df = pd.read_csv(spam_url, sep='\t', header=None, names=['label', 'text'])
spam_df['clean_text'] = spam_df['text'].apply(preprocess_text)
spam_vectorizer = TfidfVectorizer()
X_spam = spam_vectorizer.fit_transform(spam_df['clean_text'])
y_spam = spam_df['label'].map({'ham': 0, 'spam': 1})
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_spam, y_spam, test_size=0.2, random_state=42, stratify=y_spam)
spam_model = MultinomialNB()
spam_model.fit(X_train_s, y_train_s)
with open('model_spam.pkl', 'wb') as f:
    pickle.dump(spam_model, f)
with open('vectorizer_spam.pkl', 'wb') as f:
    pickle.dump(spam_vectorizer, f)


# --- Sentiment Analysis Model (using NLTK movie_reviews) ---
import nltk
from nltk.corpus import movie_reviews
nltk.download('movie_reviews')

sent_texts = []
sent_labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        sent_texts.append(' '.join(movie_reviews.words(fileid)))
        sent_labels.append(category)

sent_df = pd.DataFrame({'text': sent_texts, 'label': sent_labels})
sent_df['clean_text'] = sent_df['text'].apply(preprocess_text)
sent_vectorizer = TfidfVectorizer()
X_sent = sent_vectorizer.fit_transform(sent_df['clean_text'])
y_sent = sent_df['label']

# Check class distribution
print('Sentiment class distribution:')
print(y_sent.value_counts())

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_sent, y_sent, test_size=0.2, random_state=42, stratify=y_sent)
sent_model = LogisticRegression(max_iter=1000)
sent_model.fit(X_train_t, y_train_t)
with open('model_sentiment.pkl', 'wb') as f:
    pickle.dump(sent_model, f)
with open('vectorizer_sentiment.pkl', 'wb') as f:
    pickle.dump(sent_vectorizer, f)

print('Both models and vectorizers saved successfully.')
