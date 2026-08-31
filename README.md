# Emotion Detector

A web app that predicts the emotion behind a sentence across 28 emotion categories (27 emotions + neutral, e.g. admiration, gratitude, annoyance, fear, curiosity, sadness, joy, love...), built with scikit-learn and Streamlit.

## How it works

- Text is cleaned (lowercased, punctuation removed, stopwords removed — negation words like "not"/"never" are kept, since they matter for emotion signal)
- Cleaned text is converted into numbers using TF-IDF
- A Logistic Regression model predicts the emotion
- Trained on the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) (~37k Reddit comments, 28 emotion classes)

> **Note on class imbalance:** GoEmotions is heavily imbalanced — "neutral" alone makes up roughly a third of the data, while classes like `grief`, `pride`, and `nervousness` have only a few dozen examples. Because of this, overall accuracy can be misleading; **macro F1** and per-class precision/recall (via `classification_report`) are more representative of real performance. `class_weight='balanced'` is used during training to help offset the imbalance.

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

## Model performance

Trained on the GoEmotions dataset (28 classes, ~37k samples, highly imbalanced).

| Metric | Score |
|---|---|
| Overall accuracy | _fill in from your latest run_ |
| Macro F1 | _fill in — more meaningful given class imbalance_ |

Run `classification_report(y_test, y_pred)` after training to get the full per-class breakdown, and update this table with your final numbers.

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