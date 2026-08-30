import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
stop_words.discard("not")

EMOTION_STYLE = {
    "joy": {"emoji": "😄", "color": "#FFD93D"},
    "sadness": {"emoji": "😢", "color": "#6FA8DC"},
    "anger": {"emoji": "😡", "color": "#E06666"},
    "fear": {"emoji": "😨", "color": "#8E7CC3"},
    "love": {"emoji": "❤️", "color": "#F06292"},
    "surprise": {"emoji": "😲", "color": "#81C995"},
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


@st.cache_resource
def load_artifacts():
    with open("models/my_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/my_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/label_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return model, vectorizer, mapping


model, vectorizer, label_mapping = load_artifacts()

st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")

st.title("🎭 Emotion Detector")
st.caption("Trained on 2,000 sentences using Logistic Regression + TF-IDF")

st.divider()

examples = [
    "I can't believe I got the job, this is amazing!",
    "I miss my old friends so much.",
    "Why would they cancel the trip without telling me?",
]

col1, col2, col3 = st.columns(3)
example_clicked = None
for col, ex in zip([col1, col2, col3], examples):
    if col.button(ex[:22] + "...", use_container_width=True):
        example_clicked = ex

user_input = st.text_area(
    "Or type your own sentence:",
    value=example_clicked if example_clicked else "",
    height=100,
    placeholder="e.g. I feel like everything is finally coming together...",
)

predict_clicked = st.button("Predict emotion", type="primary", use_container_width=True)

if predict_clicked:
    if user_input.strip() == "":
        st.warning("Type a sentence first, or click one of the examples above.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 1)

        emotion = label_mapping[prediction]
        style = EMOTION_STYLE.get(emotion, {"emoji": "🤔", "color": "#CCCCCC"})

        st.markdown(
            f"""
            <div style="background-color:{style['color']}22;
                        border-left:6px solid {style['color']};
                        padding:20px; border-radius:10px; margin-top:10px;">
                <h2 style="margin:0;">{style['emoji']} {emotion.capitalize()}</h2>
                <p style="margin:4px 0 0 0; color:gray;">Confidence: {confidence}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(confidence / 100)

        with st.expander("See probability breakdown for all emotions"):
            for label_id, emo in label_mapping.items():
                prob = round(probabilities[label_id] * 100, 1)
                emo_style = EMOTION_STYLE.get(emo, {"emoji": "🤔"})
                st.write(f"{emo_style['emoji']} **{emo.capitalize()}** — {prob}%")
                st.progress(prob / 100)
