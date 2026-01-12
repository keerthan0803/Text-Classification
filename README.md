# Text Classification System

This project is a multi-task text classification system for AI/ML Engineer Intern assignments. It supports spam detection, sentiment analysis, and topic classification using Python, Flask, and scikit-learn.

## Features
- **Spam Detection**: Classifies SMS messages as spam or ham.
- **Sentiment Analysis**: Classifies movie reviews as positive or negative.
- **Topic Classification**: Classifies news articles into topics using the 20 Newsgroups dataset.
- **Web Frontend**: User-friendly interface for text input and task selection.
- **Backend API**: Flask app serving predictions for all tasks.

## Folder Structure
```
TEXT classifier/
├── app.py
├── train_and_save_models.py
├── requirements.txt
├── README.md
├── .gitignore
├── topic_label_names.pkl
├── model_spam.pkl
├── model_sentiment.pkl
├── model_topic.pkl
├── vectorizer_spam.pkl
├── vectorizer_sentiment.pkl
├── vectorizer_topic.pkl
├── data/
│   ├── sms.tsv
│   ├── movie_reviews_df.pkl
│   └── topic_20newsgroups.pkl
├── frontend/
│   ├── templates/
│   │   └── index.html
│   └── static/
└── .venv/
```

## Setup Instructions
1. **Create and activate virtual environment**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
3. **Train and save models**
   ```
   python train_and_save_models.py
   ```
4. **Run the Flask app**
   ```
   python app.py
   ```
5. **Access the frontend**
   - Open your browser at `http://127.0.0.1:5000/`

## Notes
- All datasets are stored in the `data/` folder.
- Each model and vectorizer is saved separately for each task.
- The frontend allows selection of classification task and text input.

## Requirements
See `requirements.txt` for all Python dependencies.

## License
MIT
