# Emotion Analysis

A text classification project that predicts the underlying emotion — **anger, fear, joy, love, sadness, or surprise** — from a piece of text. Several classic machine learning approaches (Logistic Regression, Naive Bayes, SVM, Random Forest) are compared using Bag-of-Words and TF-IDF features.

**Live demo:** [emotionanalysis-bip.streamlit.app](https://emotionanalysis-bip.streamlit.app)

**Best model so far:** Logistic Regression + Bag-of-Words (with class balancing) — **~89% accuracy**.

---

## Overview

Given a short sentence, the model classifies the dominant emotion being expressed. This is a supervised NLP text-classification task, trained on labeled examples and evaluated across multiple classic ML pipelines to compare how feature representation (BoW vs. TF-IDF) and model choice affect performance.

The project includes:
- A Jupyter notebook covering the full pipeline: preprocessing, feature extraction, model training, and evaluation.
- A Streamlit web app for interactive predictions, both for single sentences and batch (CSV/TXT) input.

## Features

- Text preprocessing (lowercasing, punctuation removal, stopword filtering, tokenization).
- Feature extraction with Bag-of-Words and TF-IDF.
- Multiple models trained and compared: Logistic Regression, Naive Bayes, SVM, Random Forest.
- Class-balanced training to handle uneven emotion class distribution (surprise and love are the rarest classes in the dataset).
- Interactive web app with:
  - Single-sentence prediction with confidence scores and a probability breakdown.
  - Batch prediction from an uploaded CSV or TXT file, with downloadable results.

## Model Performance

| Model                          | Features | Accuracy |
|---------------------------------|----------|----------|
| Logistic Regression (balanced) | BoW      | ~89%     |

*(Add additional rows here as more model/feature combinations are evaluated in the notebook.)*

> **Note:** Surprise and love are the rarest classes in the training data, so predictions for these two emotions are slightly less reliable than the others.

## Folder Structure

```
Emotion_analysis/
│
├── data/
│   └── train.txt                 # dataset
│
├── models/                       # generated after running the notebook
│   ├── emotion_model.pkl
│   ├── bow_vectorizer.pkl
│   └── label_encoder.pkl
│
├── notebook/
│   └── emotion_analysis.ipynb    # main notebook (preprocessing, training, evaluation)
│
├── app.py                        # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/kaflebiplob/Emotion_analysis.git
cd Emotion_analysis
pip install -r requirements.txt
```

## Usage

### 1. Train the model

Run `notebook/emotion_analysis.ipynb` from top to bottom. This preprocesses the data, trains and compares the models, and saves the best-performing pipeline to the `models/` folder (`emotion_model.pkl`, `bow_vectorizer.pkl`, `label_encoder.pkl`).

### 2. Run the app locally

Once the `models/` folder is populated, launch the Streamlit app:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` with two modes:
- **Single sentence** — type a sentence (or pick a sample) and get an instant emotion prediction with a confidence score.
- **Batch (CSV/TXT)** — upload a file of sentences and download predictions for the whole batch as a CSV.

## Tech Stack

- **Language:** Python
- **ML/NLP:** scikit-learn, NLTK
- **Data handling:** pandas, NumPy
- **App/deployment:** Streamlit, Streamlit Community Cloud

## Status & Future Work

This project is a work in progress. Planned improvements include:
- Expanding the comparison table with results from all trained models and feature combinations.
- Trying additional feature representations (e.g., word embeddings) and a neural baseline for comparison against the classic ML models.
- Improving robustness on the underrepresented emotion classes (surprise, love).

## Author

Biplob Kafle
[GitHub](https://github.com/kaflebiplob)

Feedback and suggestions are welcome — feel free to open an issue or reach out directly.