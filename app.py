import re
import joblib
import streamlit as st
from scipy.sparse import hstack

data = joblib.load("toxic_debiased.joblib")
word_vec = data["word_vec"]
char_vec = data["char_vec"]
models = data["models"]
LABELS = data["labels"]
THRESHOLD = 0.5

SAFE_TIPS = [
    ("threat", "Threats are never okay. If you're angry, step away first, then say what's bothering you without mentioning harm."),
    ("identity_hate", "Avoid attacking someone's identity or group. Focus on the specific issue instead."),
    ("severe_toxic", "This is very aggressive. Try removing the abuse and stating your actual point."),
    ("insult", "Skip the personal insult and address the idea instead."),
    ("obscene", "Try dropping the profanity. The same point lands better in calmer words."),
    ("toxic", "This may come across as rude or hostile. Try a more respectful tone."),
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict(text):
    t = clean_text(text)
    x = hstack([word_vec.transform([t]), char_vec.transform([t])]).tocsr()
    return {l: round(float(models[l].predict_proba(x)[0, 1]), 3) for l in LABELS}

st.set_page_config(page_title="Toxic Comment Detector")
st.title("Toxic Comment / Cyberbullying Detector")
st.write("Type a comment to see which toxicity categories it triggers and a suggestion for a safer way to say it.")

text = st.text_area("Paste a comment", height=100)

if st.button("Submit") and text.strip():
    scores = predict(text)
    flagged = [l for l, s in scores.items() if s >= THRESHOLD]

    st.subheader("Toxicity scores")
    for label, s in sorted(scores.items(), key=lambda x: -x[1]):
        st.write(f"**{label}**: {s*100:.0f}%")
        st.progress(s)

    if flagged:
        st.error("🚩 Flagged: " + ", ".join(flagged))
        tip = next(t for l, t in SAFE_TIPS if l in flagged)
        st.subheader("Safe reply suggestion")
        st.info(tip)
    else:
        st.success("✅ Looks safe")
