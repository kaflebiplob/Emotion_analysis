# Emotion Analysis

A text classification project that predicts emotion (anger, fear, joy, love, sadness, surprise) from text using classic ML models (Logistic Regression, Naive Bayes, SVM, Random Forest) with Bag-of-Words and TF-IDF features.

**Best model so far:** Logistic Regression + BoW (with class balancing) — ~89% accuracy.


## Setup


## Folder Structure

```
Emotion_analysis/
│
├── data/
│   └── train.txt              # dataset 
│
├── models/                    # generated after running the notebook 
│   ├── emotion_model.pkl
│   ├── bow_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebook/                    
│   ├── emotion_analysis.ipynb    # main notebook
├── requirements.txt
├── .gitignore
└── README.md
```


```bash
pip install -r requirements.txt
```

## Usage

Run `emotion_analysis.ipynb` — it trains the models and saves them to a `models/` folder.

This project is a work in progress — I'm currently experimenting with different models.

Thank you for your understanding.