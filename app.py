import re
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_META = {
    "joy": {"emoji": "😄", "color": "#F2B705"},
    "sadness": {"emoji": "😢", "color": "#4C8BF5"},
    "anger": {"emoji": "😡", "color": "#E45858"},
    "fear": {"emoji": "😨", "color": "#8E6FD1"},
    "love": {"emoji": "❤️", "color": "#F0679A"},
    "surprise": {"emoji": "😲", "color": "#5AC98C"},
}

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

.hero-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0; }
.hero-sub   { opacity: 0.65; font-size: 0.9rem; margin-top: 0.1rem; }

.result-card {
    border-radius: 10px; border: 1px solid rgba(128,128,128,0.2);
    border-left: 6px solid var(--accent); padding: 1.2rem 1.5rem;
    background: rgba(128,128,128,0.05); margin-top: 1rem;
}
.result-emotion { font-size: 1.7rem; font-weight: 800; text-transform: capitalize; margin: 0; }
.result-conf { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; opacity: 0.7; }

.legend-row {
    display: flex; justify-content: space-between; padding: 0.3rem 0;
    border-bottom: 1px dashed rgba(128,128,128,0.15); font-size: 0.85rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- LOADING ----------------
@st.cache_resource
def load_artifacts():
    with open("models/my_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/my_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/label_mapping.pkl", "rb") as f:
        mapping = pickle.load(f)
    return model, vectorizer, mapping


stop_words = set(stopwords.words("english"))
stop_words.discard("not")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


try:
    model, vectorizer, label_mapping = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model files: {e}")

# ---------------- HERO ----------------
st.markdown(
    """
<div class="hero-title">🎭 Emotion Detector</div>
<div class="hero-sub">Logistic Regression + TF-IDF · trained on 2,000 sentences</div>
""",
    unsafe_allow_html=True,
)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("**Emotion legend**")
for emo, meta in EMOTION_META.items():
    st.sidebar.markdown(
        f"<div class='legend-row'><span>{meta['emoji']} {emo.capitalize()}</span></div>",
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: Logistic Regression\nFeatures: TF-IDF (4,645 words)\nTest accuracy: ~61%"
)

page = st.sidebar.radio("Mode", ["Single sentence", "Batch (CSV/TXT)"])

# ---------------- PAGE 1: SINGLE SENTENCE ----------------
if page == "Single sentence":
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    samples = {
        "Joy": "I can't believe I got the job, this is amazing!",
        "Sadness": "I miss my old friends so much.",
        "Anger": "Why would they cancel the trip without telling me?",
        "Fear": "I feel scared and anxious about tomorrow.",
    }
    st.caption("Try an example:")
    cols = st.columns(len(samples))
    for col, (label, text) in zip(cols, samples.items()):
        if col.button(label, use_container_width=True):
            st.session_state.input_text = text

    user_input = st.text_area(
        "Enter a sentence:",
        value=st.session_state.input_text,
        height=100,
        placeholder="Type how you're feeling...",
    )

    if st.button("Predict emotion", type="primary", use_container_width=True):
        if not model_loaded:
            st.stop()
        if user_input.strip() == "":
            st.warning("Type a sentence first, or click an example above.")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            pred_id = model.predict(vector)[0]
            emotion = label_mapping[pred_id]
            confidence = round(max(probs) * 100, 1)
            meta = EMOTION_META.get(emotion, {"emoji": "🤔", "color": "#999999"})

            st.markdown(
                f"""
            <div class="result-card" style="--accent: {meta['color']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="result-emotion" style="color:{meta['color']};">
                        {meta['emoji']} {emotion}
                    </div>
                    <div class="result-conf">confidence: {confidence}%</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            probs_df = pd.DataFrame(
                {
                    "Emotion": [
                        label_mapping[i].capitalize() for i in range(len(probs))
                    ],
                    "Probability (%)": [round(p * 100, 1) for p in probs],
                }
            ).sort_values("Probability (%)", ascending=True)

            st.markdown("**Probability breakdown**")
            st.bar_chart(probs_df.set_index("Emotion"), use_container_width=True)

# ---------------- PAGE 2: BATCH ----------------
else:
    st.caption(
        "Upload a CSV with a text column, or a plain .txt file (one sentence per line)."
    )
    uploaded = st.file_uploader("Upload file", type=["csv", "txt"])

    if uploaded is not None and model_loaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded)
                text_col = st.selectbox("Which column is the text?", df_upload.columns)
                texts = df_upload[text_col].astype(str).tolist()
            else:
                texts = [
                    line.decode("utf-8").strip()
                    for line in uploaded.readlines()
                    if line.strip()
                ]
                df_upload = pd.DataFrame({"text": texts})

            st.info(f"Loaded {len(texts)} rows.")

            if st.button("Run predictions", type="primary"):
                cleaned = [clean_text(t) for t in texts]
                vectors = vectorizer.transform(cleaned)
                preds = model.predict(vectors)
                confs = model.predict_proba(vectors).max(axis=1)

                result_df = df_upload.copy()
                result_df["predicted_emotion"] = [label_mapping[p] for p in preds]
                result_df["confidence"] = np.round(confs * 100, 1)

                st.dataframe(result_df, use_container_width=True)

                st.markdown("**Emotion distribution**")
                st.bar_chart(result_df["predicted_emotion"].value_counts())

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Something went wrong reading that file: {e}")

st.divider()
st.caption("Built as a learning project · Logistic Regression on TF-IDF features")
