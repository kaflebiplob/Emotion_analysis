import re
import string
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

for pkg in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Emotions
EMOTION_META = {
    "anger": {"emoji": "😡", "color": "#EF4444"},
    "fear": {"emoji": "😨", "color": "#8B5CF6"},
    "joy": {"emoji": "😄", "color": "#F59E0B"},
    "love": {"emoji": "❤️", "color": "#EC4899"},
    "sadness": {"emoji": "😢", "color": "#3B82F6"},
    "surprise": {"emoji": "😲", "color": "#10B981"},
}

DEFAULT_META = {"emoji": "🤔", "color": "#6B7280"}

# ---------------- STYLE ----------------
st.markdown(
    """
<style>
:root {
    --border-soft: rgba(128,128,128,0.25);
}

/* ---------- Hero ---------- */
.hero-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0;
}
.hero-sub {
    opacity: 0.65;
    font-size: 0.88rem;
    margin-top: 0.15rem;
}

/* ---------- Section label ---------- */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    opacity: 0.6;
    margin-bottom: 0.4rem;
}

/* ---------- Result card ---------- */
.result-card {
    border-radius: 6px;
    border: 1px solid var(--border-soft);
    border-left: 4px solid var(--accent);
    padding: 1rem 1.3rem;
    margin-top: 1rem;
}
.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.result-emotion {
    font-size: 1.4rem;
    font-weight: 700;
    text-transform: capitalize;
    margin: 0;
}
.result-conf {
    font-family: monospace;
    font-size: 0.85rem;
    opacity: 0.7;
}

/* ---------- Breakdown label ---------- */
.breakdown-label {
    font-size: 0.85rem;
    font-weight: 600;
    opacity: 0.75;
    margin-top: 1.2rem;
    margin-bottom: 0.3rem;
}

/* ---------- Sidebar ---------- */
.sidebar-title {
    font-size: 0.8rem;
    font-weight: 600;
    opacity: 0.6;
    margin-bottom: 0.4rem;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.15rem 0;
    font-size: 0.85rem;
}
.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sidebar-card {
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 0.7rem 0.9rem;
    font-size: 0.78rem;
    line-height: 1.6;
    opacity: 0.8;
    margin-top: 0.5rem;
}

/* ---------- Footer ---------- */
.footer-caption {
    text-align: center;
    opacity: 0.5;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- LOADING ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/emotion_model.pkl")
    vectorizer = joblib.load("models/bow_vectorizer.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return model, vectorizer, encoder


stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    words = word_tokenize(text)
    cleaned = [w for w in words if w not in stop_words]
    return " ".join(cleaned)


try:
    model, vectorizer, label_encoder = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model files: {e}")

# ---------------- HERO ----------------
st.markdown(
    """
<div class="hero-title">🎭 Sentiment Analysis</div>
<div class="hero-sub">Logistic Regression (class-balanced) + Bag-of-Words</div>
""",
    unsafe_allow_html=True,
)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    '<div class="sidebar-title">Sentiments</div>', unsafe_allow_html=True
)
for emo, meta in EMOTION_META.items():
    st.sidebar.markdown(
        f"""<div class='legend-row'>
                <span class='legend-dot' style='background:{meta['color']};'></span>
                <span>{meta['emoji']} {emo.capitalize()}</span>
            </div>""",
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    """
<div class="sidebar-card">
    <b>Model</b> · Logistic Regression (class_weight='balanced')<br>
    <b>Features</b> · Bag-of-Words<br>
    <b>Classes</b> · 6 (anger, fear, joy, love, sadness, surprise)<br>
    <b>Test accuracy</b> · ~89%<br><br>
    Surprise &amp; love are the rarest classes and slightly less reliable.
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
page = st.sidebar.radio("Mode", ["Single sentence", "Batch (CSV/TXT)"])


# ---------------- PAGE 1: SINGLE SENTENCE ----------------
if page == "Single sentence":
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    samples = {
        "Anger": "This makes me so angry, I can't believe it.",
        "Fear": "I feel scared and anxious about tomorrow.",
        "Joy": "I am so happy right now, this is the best day.",
        "Love": "I feel romantic today.",
        "Sadness": "I feel really sad and lonely right now.",
        "Surprise": "I was totally shocked by the news.",
    }

    st.markdown(
        '<div class="section-label">Try an example</div>', unsafe_allow_html=True
    )
    cols = st.columns(len(samples))
    for col, (label, text) in zip(cols, samples.items()):
        if col.button(label, width="stretch"):
            st.session_state.input_text = text

    user_input = st.text_area(
        "Enter a sentence:",
        value=st.session_state.input_text,
        height=100,
        placeholder="Type how you're feeling...",
    )

    if st.button("Predict Sentiment", type="primary", width="stretch"):
        if not model_loaded:
            st.stop()
        if user_input.strip() == "":
            st.warning("Type a sentence first, or click an example above.")
        else:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            probs = model.predict_proba(vector)[0]
            pred_id = model.predict(vector)[0]
            emotion = label_encoder.inverse_transform([pred_id])[0]
            confidence = round(max(probs) * 100, 1)
            meta = EMOTION_META.get(emotion, DEFAULT_META)

            st.markdown(
                f"""
            <div class="result-card" style="--accent: {meta['color']};">
                <div class="result-top">
                    <div class="result-emotion" style="color:{meta['color']};">
                        {meta['emoji']} {emotion}
                    </div>
                    <div class="result-conf">confidence · {confidence}%</div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            probs_df = pd.DataFrame(
                {
                    "Emotion": [
                        label_encoder.classes_[i].capitalize()
                        for i in range(len(probs))
                    ],
                    "Probability (%)": [round(p * 100, 1) for p in probs],
                }
            ).sort_values("Probability (%)", ascending=False)

            st.markdown(
                '<div class="breakdown-label">Probability breakdown</div>',
                unsafe_allow_html=True,
            )

            st.bar_chart(
                probs_df.sort_values("Probability (%)", ascending=True).set_index(
                    "Emotion"
                ),
                width="stretch",
            )

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
                result_df["predicted_emotion"] = label_encoder.inverse_transform(preds)
                result_df["confidence"] = np.round(confs * 100, 1)

                st.dataframe(result_df, width="stretch")

                st.markdown(
                    '<div class="breakdown-label">Sentiment distribution</div>',
                    unsafe_allow_html=True,
                )
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
st.markdown(
    '<div class="footer-caption">Built as a learning project · Logistic Regression (class-balanced) on Bag-of-Words features</div>',
    unsafe_allow_html=True,
)
