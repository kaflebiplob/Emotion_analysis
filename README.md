# Emotion Detector

A web app that predicts the emotion behind a sentence across 28 emotion categories (27 emotions + neutral, e.g. admiration, gratitude, annoyance, fear, curiosity, sadness, joy, love...), built with scikit-learn and Streamlit.

## How it works

- Text is cleaned (lowercased, punctuation removed, stopwords removed — negation words like "not"/"never" are kept, since they matter for emotion signal)
- Cleaned text is converted into numbers using TF-IDF
- A **Linear SVM** (`LinearSVC`, wrapped in `CalibratedClassifierCV` to produce probability scores) predicts the emotion — chosen after comparing it against Logistic Regression and a Decision Tree, since it had the best accuracy and macro F1 of the three
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

Trained on the GoEmotions dataset (28 classes, ~37k samples, highly imbalanced). Three models were compared on the held-out test set; **Linear SVM** was selected as the best performer on both metrics.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 0.408 | 0.344 |
| **Linear SVM (selected)** | **0.430** | **0.359** |

Because of the severe class imbalance, macro F1 is the more meaningful number — it weighs all 28 classes equally rather than being dominated by "neutral." Run `classification_report(y_test, y_pred)` after training for the full per-class breakdown.

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