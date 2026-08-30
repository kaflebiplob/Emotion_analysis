import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
stop_words.discard("not")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


with open("models/my_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/my_vectorizer.pkl", "rb") as f:
    vectorize = pickle.load(f)

with open("models/label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

st.title("Emotion Detector")

user_input = st.text_input("Type a sentence:")

if user_input:
    cleaned = clean_text(user_input)
    vector = vectorize.transform([cleaned])
    prediction = model.predict(vector)[0]
    emotion = label_mapping[prediction]
    st.write("Predicted emotion:", emotion)
