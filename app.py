import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
stop_words.discard("not")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


def load_artifacts():
    with open("models/my_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/my_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/label_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return model, vectorizer, mapping


model, vectorizer, label_mapping = load_artifacts()

st.set_page_config(page_title="My Emotion Classifier")
st.title("What's the emotion in your sentence?")
st.caption("Trained on 2,000 samples using Logistic Regression + TF-IDF")

user_input = st.text_area("Enter a sentence:", height=100)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please type something first.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 1)

        emotion = label_mapping[prediction]
        st.subheader(f"Emotion: {emotion}")
        st.write(f"Confidence: {confidence}%")
