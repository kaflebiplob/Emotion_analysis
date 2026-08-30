# Emotion Detector

A simple web app that predicts the emotion behind a sentence (joy, sadness, anger, fear, love, or surprise), built with scikit-learn and Streamlit.

## How it works

- Text is cleaned (lowercased, punctuation removed, stopwords removed)
- Cleaned text is converted into numbers using TF-IDF
- A Logistic Regression model predicts the emotion
- Trained on 2,000 labeled sentences from Kaggle's Emotion Dataset

## Setup

1. Clone this repository
2. Create a virtual environment:
   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate
   \`\`\`
3. Install dependencies:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
4. Download NLTK stopwords (one-time, run in Python):
   \`\`\`python
   import nltk
   nltk.download('stopwords')
   \`\`\`

## Running the app

\`\`\`bash
streamlit run app.py
\`\`\`

Open your browser at `http://localhost:8501`.

## Model accuracy

~61% on a held-out test set (400 sentences), trained on 2,000 total samples.

## Project structure

\`\`\`
emotion_analysis/
├── data/
│   └── test.csv
├── models/
│   ├── my_model.pkl
│   ├── my_vectorizer.pkl
│   └── label_mapping.pkl
├── notebook/
│   └── train_my_model.ipynb
├── app.py
└── requirements.txt
\`\`\`